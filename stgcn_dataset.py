"""
Skeleton Action Recognition Dataset for ST-GCN.

Loads pre-exported .npy / .pkl data (from export_cvat_zips_to_stgcn.py).
Shape convention: (N, C=3, T=300, V=17, M=1)  where C = [x, y, confidence].

Augmentation strategy for very small datasets:
  1. Sliding window (offline): dense overlapping clips.
  2. Left-right flip (online).
  3. Gaussian noise (online).
  4. Random translation (online).
  5. Joint masking (online): randomly hide joints to improve robustness.
  6. Speed perturbation (online): temporal stretching/compression.
  7. Temporal jitter (online): small random shifts in frame indices.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------------------
# COCO 17-joint left-right pairs (0-indexed)
# ------------------------------------------------------------------------------
FLIP_PAIRS: List[Tuple[int, int]] = [
    (1, 2),   # eyes
    (3, 4),   # ears
    (5, 6),   # shoulders
    (7, 8),   # elbows
    (9, 10),  # wrists
    (11, 12), # hips
    (13, 14), # knees
    (15, 16), # ankles
]


# ------------------------------------------------------------------------------
# Offline sliding-window augmentation
# ------------------------------------------------------------------------------
def sliding_window_split(
    data: np.ndarray,
    labels: list,
    names: list,
    clip_len: int = 100,
    stride: int = 25,
    min_len: int = 30,
) -> Tuple[np.ndarray, list, list]:
    """
    Split each sequence into overlapping clips via sliding window.
    Dense stride (25) generates more clips per sequence, crucial for small datasets.
    """
    n, c, t_max, v, m = data.shape
    new_data: List[np.ndarray] = []
    new_labels: list = []
    new_names: list = []

    for i in range(n):
        seq = data[i]
        label = labels[i]
        name = names[i]

        conf = seq[2]
        frame_has_data = conf.reshape(t_max, -1).any(axis=1)
        valid_indices = np.where(frame_has_data)[0]
        if len(valid_indices) == 0:
            continue
        actual_len = int(valid_indices[-1]) + 1

        if actual_len <= clip_len:
            clip = np.zeros((c, clip_len, v, m), dtype=np.float32)
            use_len = min(actual_len, clip_len)
            clip[:, :use_len, :, :] = seq[:, :use_len, :, :]
            new_data.append(clip)
            new_labels.append(label)
            new_names.append(f"{name}_w0")
        else:
            win_id = 0
            start = 0
            while start < actual_len:
                end = start + clip_len
                clip = np.zeros((c, clip_len, v, m), dtype=np.float32)
                use_end = min(end, actual_len)
                use_len = use_end - start
                if use_len < min_len:
                    break
                clip[:, :use_len, :, :] = seq[:, start:use_end, :, :]
                new_data.append(clip)
                new_labels.append(label)
                new_names.append(f"{name}_w{win_id}")
                win_id += 1
                start += stride

    new_arr = np.stack(new_data, axis=0).astype(np.float32)
    return new_arr, new_labels, new_names


# ------------------------------------------------------------------------------
# Online augmentation transforms
# ------------------------------------------------------------------------------
def flip_lr(data: np.ndarray) -> np.ndarray:
    """Left-right flip: negate x and swap symmetric joints. data: (C, T, V, M)."""
    out = data.copy()
    out[0] = -out[0]
    for l_idx, r_idx in FLIP_PAIRS:
        out[:, :, l_idx, :], out[:, :, r_idx, :] = (
            out[:, :, r_idx, :].copy(),
            out[:, :, l_idx, :].copy(),
        )
    return out


def add_gaussian_noise(data: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Add Gaussian noise to visible joints. data: (C, T, V, M)."""
    out = data.copy()
    conf_mask = out[2:3] > 0
    noise = np.random.randn(2, *out.shape[1:]).astype(np.float32) * sigma
    out[0:2] += noise * conf_mask
    return out


def random_translation(data: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Translate skeleton by random x/y offset. data: (C, T, V, M)."""
    out = data.copy()
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    conf_mask = out[2:3] > 0
    out[0:1] += dx * conf_mask
    out[1:2] += dy * conf_mask
    return out


def random_joint_mask(data: np.ndarray, mask_ratio: float = 0.15) -> np.ndarray:
    """
    Randomly mask (zero out) some joints to improve robustness.
    Simulates partial occlusion. data: (C, T, V, M).
    """
    out = data.copy()
    c, t, v, m = out.shape
    n_mask = max(1, int(v * mask_ratio))
    joints_to_mask = np.random.choice(v, size=n_mask, replace=False)
    out[:, :, joints_to_mask, :] = 0.0
    return out


def speed_perturbation(data: np.ndarray, speed_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Temporal speed change: stretch or compress the sequence.
    Speed > 1 = faster (fewer frames used), speed < 1 = slower (more frames).
    data: (C, T, V, M).
    """
    c, t, v, m = data.shape
    speed = np.random.uniform(speed_range[0], speed_range[1])
    new_t = int(t / speed)
    new_t = max(new_t, 10)  # At least 10 frames.

    # Find actual length.
    conf = data[2]
    frame_has_data = conf.reshape(t, -1).any(axis=1)
    valid = np.where(frame_has_data)[0]
    if len(valid) == 0:
        return data.copy()
    actual_len = int(valid[-1]) + 1

    # Resample the actual portion.
    src_indices = np.linspace(0, actual_len - 1, min(new_t, t), dtype=np.float32)
    src_floor = np.floor(src_indices).astype(int)
    src_ceil = np.minimum(src_floor + 1, actual_len - 1)
    alpha = (src_indices - src_floor).reshape(1, -1, 1, 1)

    out = np.zeros((c, t, v, m), dtype=np.float32)
    interp_len = len(src_floor)
    use_len = min(interp_len, t)
    out[:, :use_len, :, :] = (
        data[:, src_floor[:use_len], :, :] * (1 - alpha[:, :use_len, :, :]) +
        data[:, src_ceil[:use_len], :, :] * alpha[:, :use_len, :, :]
    )
    return out


def temporal_jitter(data: np.ndarray, max_jitter: int = 3) -> np.ndarray:
    """
    Small random temporal perturbation: shift frame indices by small amounts.
    Preserves temporal order but adds variation. data: (C, T, V, M).
    """
    c, t, v, m = data.shape
    out = data.copy()

    # Find actual length.
    conf = data[2]
    frame_has_data = conf.reshape(t, -1).any(axis=1)
    valid = np.where(frame_has_data)[0]
    if len(valid) < 5:
        return out
    actual_len = int(valid[-1]) + 1

    # Generate jittered indices (sorted to preserve order).
    jitter = np.random.randint(-max_jitter, max_jitter + 1, size=actual_len)
    indices = np.arange(actual_len) + jitter
    indices = np.clip(indices, 0, actual_len - 1)

    out[:, :actual_len, :, :] = data[:, indices, :, :]
    return out


def random_scale(data: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Random body scale augmentation. data: (C, T, V, M)."""
    out = data.copy()
    sx = np.random.uniform(scale_range[0], scale_range[1])
    sy = np.random.uniform(scale_range[0], scale_range[1])
    conf_mask = out[2:3] > 0
    out[0:1] *= sx * conf_mask + (1 - conf_mask) * out[0:1]
    out[1:2] *= sy * conf_mask + (1 - conf_mask) * out[1:2]
    return out


def uniform_sample_frames(data: np.ndarray, clip_len: int) -> np.ndarray:
    """Uniformly sample clip_len frames. data: (C, T, V, M)."""
    c, t, v, m = data.shape
    if t == clip_len:
        return data
    if t < clip_len:
        out = np.zeros((c, clip_len, v, m), dtype=np.float32)
        out[:, :t, :, :] = data
        return out
    indices = np.linspace(0, t - 1, clip_len, dtype=int)
    return data[:, indices, :, :]


# ------------------------------------------------------------------------------
# Mixup for skeleton sequences
# ------------------------------------------------------------------------------
def mixup_data(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, float]:
    """
    Mixup two skeleton sequences. Returns mixed data and lambda.
    data1, data2: (C, T, V) tensors (already squeezed).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    mixed = lam * data1 + (1 - lam) * data2
    return mixed, lam


# ------------------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------------------
class SkeletonDataset(Dataset):
    """
    PyTorch Dataset for skeleton-based action recognition.

    Aggressive augmentation suite for very small datasets.
    """

    def __init__(
        self,
        data_path: str | Path,
        label_path: str | Path,
        clip_len: int = 100,
        is_train: bool = True,
        use_sliding_window: bool = True,
        window_stride: int = 25,
        flip_prob: float = 0.5,
        noise_sigma: float = 0.02,
        translation_max: float = 0.1,
        joint_mask_prob: float = 0.3,
        joint_mask_ratio: float = 0.15,
        speed_perturb_prob: float = 0.3,
        temporal_jitter_prob: float = 0.3,
        scale_prob: float = 0.3,
        mixup_alpha: float = 0.2,
    ):
        super().__init__()
        self.clip_len = clip_len
        self.is_train = is_train
        self.flip_prob = flip_prob
        self.noise_sigma = noise_sigma
        self.translation_max = translation_max
        self.joint_mask_prob = joint_mask_prob
        self.joint_mask_ratio = joint_mask_ratio
        self.speed_perturb_prob = speed_perturb_prob
        self.temporal_jitter_prob = temporal_jitter_prob
        self.scale_prob = scale_prob
        self.mixup_alpha = mixup_alpha

        # Load raw data.
        raw_data = np.load(str(data_path))
        with open(str(label_path), "rb") as f:
            names, labels = pickle.load(f)

        if use_sliding_window and is_train:
            self.data, self.labels, self.names = sliding_window_split(
                raw_data, labels, names,
                clip_len=clip_len,
                stride=window_stride,
                min_len=max(clip_len // 3, 30),
            )
        else:
            clips = []
            for i in range(raw_data.shape[0]):
                clip = uniform_sample_frames(raw_data[i], clip_len)
                clips.append(clip)
            self.data = np.stack(clips, axis=0).astype(np.float32)
            self.labels = list(labels)
            self.names = list(names)

        print(
            f"[SkeletonDataset] {'train' if is_train else 'val'}: "
            f"{len(self.labels)} samples, clip_len={clip_len}, "
            f"data shape={self.data.shape}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = self.data[idx].copy()  # (C, T, V, M)
        label = self.labels[idx]

        if self.is_train:
            # Speed perturbation (before other augs since it changes temporal structure).
            if random.random() < self.speed_perturb_prob:
                data = speed_perturbation(data, speed_range=(0.8, 1.2))

            # Temporal jitter.
            if random.random() < self.temporal_jitter_prob:
                data = temporal_jitter(data, max_jitter=3)

            # Left-right flip.
            if random.random() < self.flip_prob:
                data = flip_lr(data)

            # Random scale.
            if random.random() < self.scale_prob:
                data = random_scale(data, scale_range=(0.85, 1.15))

            # Translation.
            if self.translation_max > 0:
                data = random_translation(data, max_shift=self.translation_max)

            # Gaussian noise.
            if self.noise_sigma > 0:
                data = add_gaussian_noise(data, sigma=self.noise_sigma)

            # Joint masking.
            if random.random() < self.joint_mask_prob:
                data = random_joint_mask(data, mask_ratio=self.joint_mask_ratio)

        # Squeeze M dim: (C, T, V, M) -> (C, T, V)
        data = data.squeeze(-1)

        return torch.from_numpy(data), label

    def get_random_sample(self) -> Tuple[np.ndarray, int]:
        """Get a random sample (for mixup). Returns (C, T, V) array and label."""
        idx = random.randint(0, len(self) - 1)
        data = self.data[idx].copy().squeeze(-1)
        return data, self.labels[idx]


# ------------------------------------------------------------------------------
# K-Fold split utility
# ------------------------------------------------------------------------------
def create_kfold_datasets(
    data_path: str | Path,
    label_path: str | Path,
    n_folds: int = 5,
    clip_len: int = 100,
    window_stride: int = 25,
    seed: int = 42,
    **augmentation_kwargs,
) -> List[Tuple[SkeletonDataset, SkeletonDataset]]:
    """
    Create K-fold train/val dataset pairs from the combined data.

    Merges train and val data, then creates stratified K-fold splits.
    Returns list of (train_ds, val_ds) tuples.
    """
    data_dir = Path(data_path).parent

    # Load all data (merge train + val).
    train_data = np.load(str(data_dir / "train_data.npy"))
    val_data = np.load(str(data_dir / "val_data.npy"))
    all_data = np.concatenate([train_data, val_data], axis=0)

    with open(str(data_dir / "train_label.pkl"), "rb") as f:
        train_names, train_labels = pickle.load(f)
    with open(str(data_dir / "val_label.pkl"), "rb") as f:
        val_names, val_labels = pickle.load(f)

    all_names = list(train_names) + list(val_names)
    all_labels = list(train_labels) + list(val_labels)

    n_total = len(all_labels)
    print(f"[KFold] Total samples: {n_total}, creating {n_folds}-fold splits")

    # Stratified K-Fold: group by label, distribute evenly.
    rng = np.random.RandomState(seed)
    label_indices = {}
    for i, lab in enumerate(all_labels):
        label_indices.setdefault(lab, []).append(i)

    for lab in label_indices:
        rng.shuffle(label_indices[lab])

    # Assign each sample to a fold.
    fold_assignment = np.zeros(n_total, dtype=int)
    for lab, indices in label_indices.items():
        for i, idx in enumerate(indices):
            fold_assignment[idx] = i % n_folds

    fold_datasets = []
    for fold_id in range(n_folds):
        val_mask = fold_assignment == fold_id
        train_mask = ~val_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # Save temporary fold data.
        fold_dir = data_dir / f"_fold_{fold_id}"
        fold_dir.mkdir(exist_ok=True)

        fold_train_data = all_data[train_idx]
        fold_val_data = all_data[val_idx]
        fold_train_labels = [all_labels[i] for i in train_idx]
        fold_val_labels = [all_labels[i] for i in val_idx]
        fold_train_names = [all_names[i] for i in train_idx]
        fold_val_names = [all_names[i] for i in val_idx]

        np.save(fold_dir / "train_data.npy", fold_train_data)
        np.save(fold_dir / "val_data.npy", fold_val_data)
        with open(fold_dir / "train_label.pkl", "wb") as f:
            pickle.dump((fold_train_names, fold_train_labels), f)
        with open(fold_dir / "val_label.pkl", "wb") as f:
            pickle.dump((fold_val_names, fold_val_labels), f)

        train_ds = SkeletonDataset(
            fold_dir / "train_data.npy",
            fold_dir / "train_label.pkl",
            clip_len=clip_len,
            is_train=True,
            use_sliding_window=True,
            window_stride=window_stride,
            **augmentation_kwargs,
        )
        val_ds = SkeletonDataset(
            fold_dir / "val_data.npy",
            fold_dir / "val_label.pkl",
            clip_len=clip_len,
            is_train=False,
            use_sliding_window=False,
        )
        fold_datasets.append((train_ds, val_ds))

        print(f"  Fold {fold_id}: train={len(train_ds)} val={len(val_ds)} "
              f"(train_labels={dict(sorted(zip(*np.unique(fold_train_labels, return_counts=True))))}, "
              f"val_labels={dict(sorted(zip(*np.unique(fold_val_labels, return_counts=True))))})")

    return fold_datasets


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "stgcn_action4"
    ds = SkeletonDataset(
        data_dir / "train_data.npy",
        data_dir / "train_label.pkl",
        clip_len=100,
        is_train=True,
        use_sliding_window=True,
        window_stride=25,
    )
    print(f"Total train clips (stride=25): {len(ds)}")
    x, y = ds[0]
    print(f"Sample shape: {x.shape}, label: {y}")
