"""
Standalone ST-GCN (Spatial-Temporal Graph Convolutional Network) for skeleton-based
action recognition using COCO 17-joint layout.

Architecture reference:
  Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based
  Action Recognition", AAAI 2018.

This implementation matches the MMAction2/pyskl weight format so pretrained
checkpoints can be loaded directly. Self-contained (no mmaction2/mmengine).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# COCO 17-joint skeleton graph
# ==============================================================================
#  0:nose       1:l_eye     2:r_eye     3:l_ear     4:r_ear
#  5:l_shoulder 6:r_shoulder 7:l_elbow  8:r_elbow   9:l_wrist
# 10:r_wrist   11:l_hip    12:r_hip    13:l_knee   14:r_knee
# 15:l_ankle   16:r_ankle

NUM_JOINTS = 17

# Edges (undirected).
COCO_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2),     # nose -> eyes
    (1, 3), (2, 4),     # eyes -> ears
    (0, 5), (0, 6),     # nose -> shoulders  (neck approximation)
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # shoulders -> hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
    (11, 12),           # hip-to-hip
]

CENTER_JOINT = 0


def build_adjacency_matrix() -> np.ndarray:
    """Build symmetric adjacency matrix with self-loops for COCO 17-joint skeleton."""
    adj = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for i, j in COCO_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


def build_spatial_partition(adj: np.ndarray, center: int = CENTER_JOINT) -> np.ndarray:
    """
    Distance-based spatial partitioning (3 subsets per Yan et al.):
      0 = self-loop
      1 = centripetal (closer to center)
      2 = centrifugal (farther from center)
    Returns: (3, V, V).
    """
    v = adj.shape[0]
    dist = np.full(v, np.inf)
    dist[center] = 0
    queue = [center]
    visited = {center}
    while queue:
        node = queue.pop(0)
        for neighbor in range(v):
            if adj[node, neighbor] > 0 and neighbor not in visited:
                dist[neighbor] = dist[node] + 1
                visited.add(neighbor)
                queue.append(neighbor)

    partition = np.zeros((3, v, v), dtype=np.float32)
    for i in range(v):
        for j in range(v):
            if adj[i, j] == 0:
                continue
            if i == j:
                partition[0, i, j] = 1.0
            elif dist[j] <= dist[i]:
                partition[1, i, j] = 1.0
            else:
                partition[2, i, j] = 1.0

    return partition


def normalize_adjacency(adj_subset: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    d = adj_subset.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, np.power(d, -0.5), 0.0)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ adj_subset @ d_mat


# ==============================================================================
# Graph Convolution (Conv1x1 style -- matches MMAction2 weight format)
# ==============================================================================
class SpatialGraphConv(nn.Module):
    """
    Spatial GCN using Conv2d(1x1) per partition subset.

    MMAction2 convention:
      - A: fixed normalized adjacency (3, V, V) -- buffer (not learnable)
      - PA: learnable partition-aware adjacency (3, V, V) -- parameter
      - conv: Conv2d(in_ch, out_ch * K, kernel_size=1)
      - bn: BatchNorm2d(out_ch)
    """

    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray):
        """
        Args:
            in_channels: input channels.
            out_channels: output channels.
            A: adjacency partition matrix of shape (K, V, V).
        """
        super().__init__()
        K = A.shape[0]  # 3
        self.K = K
        self.out_channels = out_channels

        # Fixed adjacency (buffer).
        self.register_buffer("A", torch.from_numpy(A).float())
        # Learnable adjacency residual (PA in MMAction2).
        self.PA = nn.Parameter(torch.from_numpy(A).float().clone())

        # Conv1x1: maps (N, C_in, T, V) -> (N, C_out*K, T, V)
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C_in, T, V) -> (N, C_out, T, V)"""
        n, c, t, v = x.shape

        # Combined adjacency = fixed + learnable.
        A_combined = self.A + self.PA  # (K, V, V)

        # Conv1x1: (N, C_in, T, V) -> (N, C_out*K, T, V)
        x_conv = self.conv(x)  # (N, C_out*K, T, V)

        # Reshape to (N, K, C_out, T, V)
        x_conv = x_conv.reshape(n, self.K, self.out_channels, t, v)

        # Graph aggregation: for each partition k, multiply by adjacency.
        # x_conv[:, k]: (N, C_out, T, V)
        # A_combined[k]: (V, V)
        # Result: sum over k of (x_conv[:, k] @ A_combined[k])
        out = torch.zeros(n, self.out_channels, t, v, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            # (N, C_out, T, V) @ (V, V) -> (N, C_out, T, V)
            out = out + torch.einsum("nctv,vw->nctw", x_conv[:, k], A_combined[k])

        out = self.bn(out)
        return out


# ==============================================================================
# ST-GCN Block
# ==============================================================================
class STGCNBlock(nn.Module):
    """
    One ST-GCN block: SpatialGCN -> ReLU -> TemporalConv -> ReLU + Residual.
    Matches MMAction2 `STGCNBlock` weight layout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: np.ndarray,
        stride: int = 1,
        dropout: float = 0.0,
        temporal_kernel_size: int = 9,
    ):
        super().__init__()

        # Spatial GCN (with built-in BN).
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)

        # Temporal convolution.
        pad = (temporal_kernel_size - 1) // 2
        self.tcn = nn.Module()
        self.tcn.conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel_size, 1),
            stride=(stride, 1),
            padding=(pad, 0),
            bias=True,
        )
        self.tcn.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual connection.
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Module()
            self.residual.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=(stride, 1), bias=True,
            )
            self.residual.bn = nn.BatchNorm2d(out_channels)
            self._has_residual_transform = True
        else:
            self.residual = nn.Identity()
            self._has_residual_transform = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T, V) -> (N, C', T', V)"""
        # Residual.
        if self._has_residual_transform:
            res = self.residual.bn(self.residual.conv(x))
        else:
            res = self.residual(x)

        # Spatial GCN (includes BN).
        out = self.gcn(x)
        out = self.relu(out)

        # Temporal Conv.
        out = self.tcn.bn(self.tcn.conv(out))
        out = self.dropout(out)

        # Residual + activation.
        out = self.relu(out + res)
        return out


# ==============================================================================
# Full ST-GCN Model
# ==============================================================================
class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network.

    10 ST-GCN blocks with channel progression:
        3 -> 64 -> 64 -> 64 -> 64 -> 128 -> 128 -> 128 -> 256 -> 256 -> 256
    Temporal stride 2 at blocks that increase channels.

    Weight naming matches MMAction2:
      backbone.data_bn -> data_bn
      backbone.gcn.{i}.gcn.{A,PA,conv,bn} -> gcn.{i}.gcn.{A,PA,conv,bn}
      backbone.gcn.{i}.tcn.{conv,bn} -> gcn.{i}.tcn.{conv,bn}
      backbone.gcn.{i}.residual.{conv,bn} -> gcn.{i}.residual.{conv,bn}
    """

    CHANNEL_CONFIG = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Build graph.
        adj = build_adjacency_matrix()
        A = build_spatial_partition(adj, CENTER_JOINT)

        # Normalize each partition separately.
        A_norm = np.stack([normalize_adjacency(A[k]) for k in range(A.shape[0])], axis=0)

        # Input batch normalization.
        # MMAction2 uses (in_ch * V) = 3 * 17 = 51
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)

        # ST-GCN blocks -- named "gcn" to match MMAction2 key prefix.
        channels = [in_channels] + self.CHANNEL_CONFIG
        self.gcn = nn.ModuleList()
        for i in range(len(self.CHANNEL_CONFIG)):
            c_in = channels[i]
            c_out = channels[i + 1]
            stride = 2 if (c_in != c_out and c_in < c_out) else 1
            self.gcn.append(
                STGCNBlock(c_in, c_out, A_norm, stride=stride, dropout=dropout)
            )

        # Classification head.
        self.fc = nn.Linear(self.CHANNEL_CONFIG[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C=3, T, V=17)
        returns: (N, num_classes)
        """
        n, c, t, v = x.shape

        # Input normalization: (N, C, T, V) -> (N, C*V, T) for BN1d.
        x_bn = x.permute(0, 3, 1, 2).reshape(n, v * c, t)
        x_bn = self.data_bn(x_bn)
        x = x_bn.reshape(n, v, c, t).permute(0, 2, 3, 1)  # (N, C, T, V)

        # ST-GCN blocks.
        for block in self.gcn:
            x = block(x)

        # Global average pooling.
        x = x.mean(dim=[2, 3])  # (N, C_last)

        # Classification.
        x = self.fc(x)
        return x


# ==============================================================================
# Pretrained weight loading
# ==============================================================================
def load_pretrained_backbone(
    model: STGCN,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load pretrained weights from an MMAction2 ST-GCN checkpoint.

    Maps MMAction2 keys (backbone.xxx) to our model keys by stripping 'backbone.'.
    The classification head (cls_head.fc) is skipped since num_classes differs.

    Returns:
        (matched_keys, skipped_keys)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        src_state = ckpt["state_dict"]
    else:
        src_state = ckpt

    tgt_state = model.state_dict()
    matched_keys: List[str] = []
    skipped_keys: List[str] = []

    # Build mapping: strip 'backbone.' from source keys.
    src_cleaned: Dict[str, torch.Tensor] = {}
    for key, val in src_state.items():
        clean = key
        for prefix in ["backbone.", "module.", "model."]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        # Skip classification head from source.
        if "cls_head" in key or "fc." in clean:
            continue
        src_cleaned[clean] = val

    new_state: Dict[str, torch.Tensor] = {}
    for tgt_key, tgt_val in tgt_state.items():
        # Skip our classification head.
        if "fc." in tgt_key:
            skipped_keys.append(tgt_key)
            continue

        if tgt_key in src_cleaned and src_cleaned[tgt_key].shape == tgt_val.shape:
            new_state[tgt_key] = src_cleaned[tgt_key]
            matched_keys.append(tgt_key)
        else:
            skipped_keys.append(tgt_key)

    model.load_state_dict(new_state, strict=False)

    if verbose:
        print(f"[Pretrained] Matched {len(matched_keys)}/{len(tgt_state)} keys.")
        if matched_keys:
            print(f"[Pretrained] Sample matched: {matched_keys[:5]}")
        if skipped_keys:
            print(f"[Pretrained] Skipped: {skipped_keys[:5]}{'...' if len(skipped_keys)>5 else ''}")

    return matched_keys, skipped_keys


# ==============================================================================
# Quick test
# ==============================================================================
if __name__ == "__main__":
    model = STGCN(in_channels=3, num_classes=4, dropout=0.3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print model keys for debugging.
    print("\nModel state_dict keys (first 15):")
    for i, k in enumerate(model.state_dict().keys()):
        if i >= 15:
            print("  ...")
            break
        print(f"  {k}: {tuple(model.state_dict()[k].shape)}")

    # Dummy forward pass.
    x = torch.randn(4, 3, 100, 17)
    out = model(x)
    print(f"\nInput: {x.shape} -> Output: {out.shape}")
    assert out.shape == (4, 4), f"Expected (4,4), got {out.shape}"
    print("OK -- model forward pass works.")
