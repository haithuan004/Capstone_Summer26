"""
Training & Evaluation script for ST-GCN skeleton action recognition.

Anti-overfitting features:
  - K-Fold cross-validation (use all data, no waste)
  - 2-phase training: freeze backbone -> train head -> unfreeze all
  - Mixup regularization
  - Heavy dropout (0.5) and weight decay (5e-4)
  - Gradient clipping
  - Label smoothing
  - Class-weighted loss
  - Full WandB integration

Usage:
    python train_stgcn.py                              # K-fold + wandb
    python train_stgcn.py --no-kfold                   # single split
    python train_stgcn.py --no-wandb --epochs 200      # no wandb
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from stgcn_dataset import SkeletonDataset, create_kfold_datasets, mixup_data
from stgcn_model import STGCN, load_pretrained_backbone

# ------------------------------------------------------------------------------
# Action class names
# ------------------------------------------------------------------------------
ACTION_CLASSES = ("standing", "walking", "sitting", "falling")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Data.
    p.add_argument("--data-dir", type=Path, default=Path("stgcn_action4"))
    p.add_argument("--clip-len", type=int, default=100)
    p.add_argument("--window-stride", type=int, default=25,
                    help="Dense stride for more sliding window clips.")

    # Model.
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--dropout", type=float, default=0.5,
                    help="High dropout for small dataset.")
    p.add_argument("--num-classes", type=int, default=4)

    # Training.
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4,
                    help="Strong weight decay to reduce overfitting.")
    p.add_argument("--label-smoothing", type=float, default=0.2,
                    help="Higher label smoothing for small dataset.")
    p.add_argument("--early-stop-patience", type=int, default=30)
    
    # Imbalance handling.
    p.add_argument("--loss-type", type=str, default="focal", choices=["ce", "focal"],
                    help="Loss function type: 'ce' or 'focal'.")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--oversample", action="store_true",
                    help="Use WeightedRandomSampler to oversample minority classes.")

    # 2-phase training: freeze backbone.
    p.add_argument("--freeze-epochs", type=int, default=15,
                    help="Epochs to freeze backbone (train head only). 0 = no freeze.")
    p.add_argument("--freeze-lr", type=float, default=1e-3,
                    help="Higher LR during freeze phase (head only).")

    # Mixup.
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                    help="Mixup alpha (0 = disabled).")

    # K-fold.
    p.add_argument("--no-kfold", action="store_true",
                    help="Disable K-fold, use original train/val split.")
    p.add_argument("--n-folds", type=int, default=5)

    # Augmentation.
    p.add_argument("--flip-prob", type=float, default=0.5)
    p.add_argument("--noise-sigma", type=float, default=0.02)
    p.add_argument("--translation-max", type=float, default=0.1)
    p.add_argument("--joint-mask-prob", type=float, default=0.3)
    p.add_argument("--speed-perturb-prob", type=float, default=0.3)

    # WandB.
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="stgcn-action4")
    p.add_argument("--wandb-run-name", type=str, default=None)

    # Output.
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints"))

    return p.parse_args()


# ------------------------------------------------------------------------------
# Loss definitions
# ------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ------------------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------------------
def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int,
) -> Dict:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(all_preds, all_labels):
        cm[true, pred] += 1

    correct = sum(1 for p, t in zip(all_preds, all_labels) if p == t)
    accuracy = correct / max(len(all_labels), 1)

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class[ACTION_CLASSES[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[c, :].sum()),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        n = counts.get(c, 1)
        weights.append(total / (num_classes * max(n, 1)))
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean()
    return w


# ------------------------------------------------------------------------------
# Freeze / unfreeze backbone utilities
# ------------------------------------------------------------------------------
def freeze_backbone(model: STGCN) -> int:
    """Freeze all layers except the classification head. Returns count frozen."""
    frozen = 0
    for name, param in model.named_parameters():
        if "fc." not in name:
            param.requires_grad = False
            frozen += 1
        else:
            param.requires_grad = True
    return frozen


def unfreeze_all(model: STGCN) -> int:
    """Unfreeze all parameters. Returns count unfrozen."""
    unfrozen = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen += 1
    return unfrozen


# ------------------------------------------------------------------------------
# Training loop with mixup
# ------------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mixup_alpha: float = 0.0,
    dataset: Optional[SkeletonDataset] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            # Mixup.
            if mixup_alpha > 0 and dataset is not None:
                # In-batch mixup: shuffle batch.
                perm = torch.randperm(data.size(0), device=device)
                data2 = data[perm]
                labels2 = labels[perm]

                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5
                mixed_data = lam * data + (1 - lam) * data2

                logits = model(mixed_data)
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels2)

                preds = logits.argmax(dim=1)
                correct += (lam * (preds == labels).float() + (1 - lam) * (preds == labels2).float()).sum().item()
            else:
                logits = model(data)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.size(0)
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for data, labels in loader:
        data = data.to(device, non_blocking=True)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(data)
            loss = criterion(logits, labels)

        total_loss += loss.item() * data.size(0)
        total += data.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    metrics = compute_metrics(all_preds, all_labels, num_classes)
    return avg_loss, metrics


# ------------------------------------------------------------------------------
# Train one fold
# ------------------------------------------------------------------------------
def train_fold(
    fold_id: int,
    train_ds: SkeletonDataset,
    val_ds: SkeletonDataset,
    args: argparse.Namespace,
    device: torch.device,
    wandb_run=None,
) -> Dict:
    """Train one fold and return best metrics."""

    if args.oversample:
        labels = train_ds.labels
        counts = Counter(labels)
        weights = [1.0 / counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Fresh model per fold.
    model = STGCN(
        in_channels=3,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    # Load pretrained weights.
    if args.pretrained and Path(args.pretrained).exists():
        load_pretrained_backbone(model, args.pretrained, verbose=(fold_id == 0))

    # Class-weighted loss.
    class_weights = compute_class_weights(train_ds.labels, args.num_classes).to(device)
    if args.loss_type == "focal":
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # -- Phase 1: Freeze backbone, train head only --
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"best_model_fold{fold_id}.pth"

    scaler = GradScaler(enabled=(device.type == "cuda"))
    total_epochs = args.epochs

    if args.freeze_epochs > 0 and args.pretrained:
        n_frozen = freeze_backbone(model)
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=args.freeze_lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.freeze_epochs, eta_min=1e-5)

        print(f"\n  [Phase 1] Freeze backbone ({n_frozen} params frozen), train head for {args.freeze_epochs} epochs")
        for epoch in range(1, args.freeze_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                mixup_alpha=0,  # No mixup during head warmup.
            )
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
            scheduler.step()
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]
            lr = scheduler.get_last_lr()[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict(), "val_metrics": val_metrics}, ckpt_path)

            prefix = f"fold{fold_id}/" if not args.no_kfold else ""
            print(f"  F{fold_id} Phase1 Ep {epoch:3d}/{args.freeze_epochs} | "
                  f"tl={train_loss:.4f} ta={train_acc:.4f} | "
                  f"vl={val_loss:.4f} va={val_acc:.4f} vf1={val_f1:.4f} | lr={lr:.2e}")

            if wandb_run:
                wandb_run.log({
                    f"{prefix}train/loss": train_loss,
                    f"{prefix}train/accuracy": train_acc,
                    f"{prefix}val/loss": val_loss,
                    f"{prefix}val/accuracy": val_acc,
                    f"{prefix}val/macro_f1": val_f1,
                    f"{prefix}phase": 1,
                    f"{prefix}lr": lr,
                })

        # Unfreeze for phase 2.
        n_unfrozen = unfreeze_all(model)
        print(f"  [Phase 2] Unfreeze all ({n_unfrozen} params), fine-tune full model")

    # -- Phase 2: Full fine-tuning with all regularization --
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    remaining_epochs = total_epochs - args.freeze_epochs if args.pretrained else total_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)
    patience_counter = 0

    for epoch in range(1, remaining_epochs + 1):
        global_epoch = epoch + (args.freeze_epochs if args.pretrained else 0)
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            mixup_alpha=args.mixup_alpha,
            dataset=train_ds,
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
        scheduler.step()

        elapsed = time.time() - t0
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        lr = scheduler.get_last_lr()[0]

        print(
            f"  F{fold_id} Ep {global_epoch:3d}/{total_epochs} | "
            f"tl={train_loss:.4f} ta={train_acc:.4f} | "
            f"vl={val_loss:.4f} va={val_acc:.4f} vf1={val_f1:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
            + (" *" if val_acc > best_val_acc else "")
        )

        if wandb_run:
            prefix = f"fold{fold_id}/" if not args.no_kfold else ""
            log_dict = {
                f"{prefix}epoch": global_epoch,
                f"{prefix}train/loss": train_loss,
                f"{prefix}train/accuracy": train_acc,
                f"{prefix}val/loss": val_loss,
                f"{prefix}val/accuracy": val_acc,
                f"{prefix}val/macro_f1": val_f1,
                f"{prefix}val/macro_precision": val_metrics["macro_precision"],
                f"{prefix}val/macro_recall": val_metrics["macro_recall"],
                f"{prefix}phase": 2,
                f"{prefix}lr": lr,
            }
            for cls_name, cls_m in val_metrics["per_class"].items():
                log_dict[f"{prefix}val/{cls_name}/f1"] = cls_m["f1"]

            if epoch % 10 == 0 or epoch == remaining_epochs:
                import wandb
                cm = val_metrics["confusion_matrix"]
                cm_table = wandb.Table(
                    columns=[""] + list(ACTION_CLASSES),
                    data=[
                        [ACTION_CLASSES[i]] + [cm[i][j] for j in range(args.num_classes)]
                        for i in range(args.num_classes)
                    ],
                )
                log_dict[f"{prefix}val/confusion_matrix"] = cm_table

            wandb_run.log(log_dict)

        is_best = val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1)
        if is_best:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = global_epoch
            patience_counter = 0
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_metrics": val_metrics,
            }, ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"  [Early Stop] No improvement for {args.early_stop_patience} epochs.")
            break

    # Load best model and do final eval.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)

    print(f"  Fold {fold_id} Best: acc={best_val_acc:.4f} f1={best_val_f1:.4f} @ epoch {best_epoch}")
    return final_metrics


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # WandB.
    wandb_run = None
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or (
                f"kfold{args.n_folds}-freeze{args.freeze_epochs}-drop{args.dropout}-mixup{args.mixup_alpha}"
            )
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
            )
            print(f"[WandB] Initialized: {run_name}")
        except Exception as e:
            print(f"[WandB] Init failed: {e}")
            wandb_run = None

    augmentation_kwargs = dict(
        flip_prob=args.flip_prob,
        noise_sigma=args.noise_sigma,
        translation_max=args.translation_max,
        joint_mask_prob=args.joint_mask_prob,
        speed_perturb_prob=args.speed_perturb_prob,
    )

    if args.no_kfold:
        # Single split mode.
        data_dir = args.data_dir.resolve()
        train_ds = SkeletonDataset(
            data_dir / "train_data.npy", data_dir / "train_label.pkl",
            clip_len=args.clip_len, is_train=True,
            use_sliding_window=True, window_stride=args.window_stride,
            **augmentation_kwargs,
        )
        val_ds = SkeletonDataset(
            data_dir / "val_data.npy", data_dir / "val_label.pkl",
            clip_len=args.clip_len, is_train=False, use_sliding_window=False,
        )
        metrics = train_fold(0, train_ds, val_ds, args, device, wandb_run)
        print_final_report([metrics], args.num_classes)
    else:
        # K-Fold cross-validation.
        data_dir = args.data_dir.resolve()
        fold_datasets = create_kfold_datasets(
            data_dir / "train_data.npy",
            data_dir / "train_label.pkl",
            n_folds=args.n_folds,
            clip_len=args.clip_len,
            window_stride=args.window_stride,
            **augmentation_kwargs,
        )

        all_metrics = []
        for fold_id, (train_ds, val_ds) in enumerate(fold_datasets):
            print(f"\n{'='*70}")
            print(f"FOLD {fold_id + 1}/{args.n_folds}")
            print(f"{'='*70}")
            metrics = train_fold(fold_id, train_ds, val_ds, args, device, wandb_run)
            all_metrics.append(metrics)

        print_final_report(all_metrics, args.num_classes)

        if wandb_run:
            import wandb
            # Log aggregated K-fold metrics.
            accs = [m["accuracy"] for m in all_metrics]
            f1s = [m["macro_f1"] for m in all_metrics]
            wandb_run.summary["kfold/mean_accuracy"] = np.mean(accs)
            wandb_run.summary["kfold/std_accuracy"] = np.std(accs)
            wandb_run.summary["kfold/mean_f1"] = np.mean(f1s)
            wandb_run.summary["kfold/std_f1"] = np.std(f1s)

    if wandb_run:
        wandb_run.finish()

    return 0


def print_final_report(all_metrics: List[Dict], num_classes: int):
    """Print aggregated results from all folds."""
    n = len(all_metrics)
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS ({n} fold{'s' if n > 1 else ''})")
    print(f"{'='*70}")

    accs = [m["accuracy"] for m in all_metrics]
    f1s = [m["macro_f1"] for m in all_metrics]
    precs = [m["macro_precision"] for m in all_metrics]
    recs = [m["macro_recall"] for m in all_metrics]

    if n > 1:
        print(f"  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (folds: {[f'{a:.3f}' for a in accs]})")
        print(f"  Macro F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  (folds: {[f'{f:.3f}' for f in f1s]})")
        print(f"  Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
        print(f"  Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    else:
        m = all_metrics[0]
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Macro F1:  {m['macro_f1']:.4f}")
        print(f"  Precision: {m['macro_precision']:.4f}")
        print(f"  Recall:    {m['macro_recall']:.4f}")

    # Per-class aggregated metrics.
    print(f"\n  Per-class (mean across folds):")
    for cls_name in ACTION_CLASSES:
        p = np.mean([m["per_class"][cls_name]["precision"] for m in all_metrics])
        r = np.mean([m["per_class"][cls_name]["recall"] for m in all_metrics])
        f1 = np.mean([m["per_class"][cls_name]["f1"] for m in all_metrics])
        print(f"    {cls_name:12s} P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    # Aggregated confusion matrix.
    if n > 1:
        total_cm = np.zeros((num_classes, num_classes), dtype=int)
        for m in all_metrics:
            total_cm += np.array(m["confusion_matrix"])
        print(f"\n  Aggregated Confusion Matrix (rows=true, cols=pred):")
        header = "            " + "".join(f"{c:>10s}" for c in ACTION_CLASSES)
        print(header)
        for i, cls in enumerate(ACTION_CLASSES):
            row = f"  {cls:>10s}" + "".join(f"{total_cm[i][j]:>10d}" for j in range(num_classes))
            print(row)

    # Save results.
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    results = {
        "n_folds": n,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "per_fold": all_metrics,
    }
    with open(save_dir / "kfold_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_dir / 'kfold_results.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
