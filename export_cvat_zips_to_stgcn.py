from __future__ import annotations

"""
Export CVAT skeleton zips (interpolated or raw) to ST-GCN numpy format for mmskeleton.

Output (per split):
  - train_data.npy, val_data.npy  shape (N, 3, T, 17, 1)  float32
  - train_label.pkl, val_label.pkl  (sample_names, labels)  labels int 0..num_classes-1

Channel 0,1: x, y (normalized per frame: hip center + torso scale).
Channel 2: confidence (1 visible, 0 missing/outside).

Classes from skeleton attribute name='action': standing, walking, sitting, falling.
"""

import argparse
import pickle
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Must match mmskeleton graph.py layout 'coco' (17 joints, 0-indexed COCO order).
COCO_LABELS: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
LABEL_TO_IDX = {n: i for i, n in enumerate(COCO_LABELS)}

ACTION_CLASSES = ("standing", "walking", "sitting", "falling")
CLASS_TO_ID = {a: i for i, a in enumerate(ACTION_CLASSES)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zip-dir",
        type=Path,
        default=Path("interpolated_zips"),
        help="Folder containing *.zip CVAT exports.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stgcn_action4"),
        help="Output directory for .npy and .pkl files.",
    )
    p.add_argument(
        "--max-frame",
        type=int,
        default=300,
        help="Max sequence length (pad/truncate).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction (0-1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split RNG seed.",
    )
    return p.parse_args()


def parse_xy(text: str | None) -> Tuple[float, float] | None:
    if not text:
        return None
    parts = [x.strip() for x in text.split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def collect_action_from_skeleton(sk: ET.Element) -> str | None:
    for attr in sk.findall("./attribute"):
        if attr.get("name") != "action":
            continue
        if attr.text is None:
            return None
        a = attr.text.strip().lower()
        if a in CLASS_TO_ID:
            return a
    return None


def skeleton_to_xy_conf(sk: ET.Element) -> np.ndarray:
    """Shape (17, 3): x, y, conf."""
    out = np.zeros((17, 3), dtype=np.float32)
    for pt in sk.findall("./points"):
        lab = pt.get("label")
        if lab not in LABEL_TO_IDX:
            continue
        idx = LABEL_TO_IDX[lab]
        outside = pt.get("outside") == "1"
        xy = parse_xy(pt.get("points"))
        if xy is None or outside:
            out[idx] = [0.0, 0.0, 0.0]
        else:
            out[idx] = [xy[0], xy[1], 1.0]
    return out


def majority_label_from_track(track: ET.Element) -> int | None:
    actions: List[int] = []
    for sk in track.findall("./skeleton"):
        a = collect_action_from_skeleton(sk)
        if a is not None:
            actions.append(CLASS_TO_ID[a])
    if not actions:
        return None
    return Counter(actions).most_common(1)[0][0]


def track_to_tensor(track: ET.Element, max_t: int) -> Tuple[np.ndarray | None, int | None]:
    frame_map: Dict[int, ET.Element] = {}
    for sk in track.findall("./skeleton"):
        ft = sk.get("frame")
        if ft is None or not str(ft).isdigit():
            continue
        frame_map[int(ft)] = sk
    if not frame_map:
        return None, None

    label = majority_label_from_track(track)
    if label is None:
        return None, None

    frames = sorted(frame_map.keys())
    if len(frames) > max_t:
        frames = frames[:max_t]
    t_len = len(frames)
    raw = np.zeros((3, t_len, 17, 1), dtype=np.float32)
    for ti, f in enumerate(frames):
        xyv = skeleton_to_xy_conf(frame_map[f])
        raw[0, ti, :, 0] = xyv[:, 0]
        raw[1, ti, :, 0] = xyv[:, 1]
        raw[2, ti, :, 0] = xyv[:, 2]

    normalize_sequence_inplace(raw)
    return raw, label


def image_list_to_tensor(root: ET.Element, max_t: int) -> Tuple[np.ndarray | None, int | None]:
    images = root.findall("./image")
    if not images:
        return None, None

    def sort_key(img: ET.Element) -> int:
        i = img.get("id") or img.get("name") or "0"
        try:
            return int(str(i).split(".")[0])
        except ValueError:
            return hash(str(i)) % (10**9)

    images = sorted(images, key=sort_key)
    if len(images) > max_t:
        images = images[:max_t]

    actions: List[int] = []
    raw = np.zeros((3, len(images), 17, 1), dtype=np.float32)
    for ti, img in enumerate(images):
        sks = img.findall("./skeleton")
        if not sks:
            continue
        xyv = skeleton_to_xy_conf(sks[0])
        raw[0, ti, :, 0] = xyv[:, 0]
        raw[1, ti, :, 0] = xyv[:, 1]
        raw[2, ti, :, 0] = xyv[:, 2]
        a = collect_action_from_skeleton(sks[0])
        if a is not None:
            actions.append(CLASS_TO_ID[a])

    if not np.any(raw[2] > 0):
        return None, None
    label = Counter(actions).most_common(1)[0][0] if actions else None
    if label is None:
        return None, None
    normalize_sequence_inplace(raw)
    return raw, label


def normalize_sequence_inplace(data: np.ndarray) -> None:
    """data shape (3, T, V, 1). Per-frame: center on hips/shoulders, scale by body size."""
    _, t, _, _ = data.shape
    ls, rs, lh, rh = 5, 6, 11, 12
    for ti in range(t):
        conf = data[2, ti, :, 0]
        xy = data[0:2, ti, :, 0].copy()

        shoulder_w = np.linalg.norm(xy[:, ls] - xy[:, rs]) if conf[ls] > 0 and conf[rs] > 0 else 0.0
        hip_w = np.linalg.norm(xy[:, lh] - xy[:, rh]) if conf[lh] > 0 and conf[rh] > 0 else 0.0
        torso = 0.0
        if conf[ls] > 0 and conf[rs] > 0 and conf[lh] > 0 and conf[rh] > 0:
            sm = (xy[:, ls] + xy[:, rs]) * 0.5
            hm = (xy[:, lh] + xy[:, rh]) * 0.5
            torso = float(np.linalg.norm(sm - hm))
        scale = max(shoulder_w, hip_w, torso, 1e-3)

        if conf[lh] > 0 and conf[rh] > 0:
            c = (xy[:, lh] + xy[:, rh]) * 0.5
        elif conf[ls] > 0 and conf[rs] > 0:
            c = (xy[:, ls] + xy[:, rs]) * 0.5
        elif conf[0] > 0:
            c = xy[:, 0]
        else:
            vis = conf > 0
            if not np.any(vis):
                continue
            c = xy[:, vis].mean(axis=1)

        data[0:2, ti, :, 0] = (xy - c.reshape(2, 1)) / scale


def pad_to_max_time(
    seq: np.ndarray, max_t: int
) -> np.ndarray:
    """seq (3, T, 17, 1) -> (3, max_t, 17, 1)."""
    c, t, v, m = seq.shape
    if t == max_t:
        return seq
    out = np.zeros((c, max_t, v, m), dtype=np.float32)
    out[:, :t, :, :] = seq
    return out


def read_xml_root(zpath: Path) -> ET.Element:
    with zipfile.ZipFile(zpath, "r") as zf:
        xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
        if xml_name is None:
            raise FileNotFoundError(f"No XML in {zpath.name}")
        with zf.open(xml_name) as f:
            return ET.parse(f).getroot()


def extract_samples(zip_path: Path, max_t: int) -> List[Tuple[str, np.ndarray, int]]:
    root = read_xml_root(zip_path)
    out: List[Tuple[str, np.ndarray, int]] = []
    tracks = root.findall("./track")
    base = zip_path.stem

    if tracks:
        for ti, track in enumerate(tracks):
            arr, lab = track_to_tensor(track, max_t)
            if arr is None or lab is None:
                continue
            name = f"{base}_track{ti}"
            out.append((name, pad_to_max_time(arr, max_t), lab))
    else:
        arr, lab = image_list_to_tensor(root, max_t)
        if arr is not None and lab is not None:
            out.append((base, pad_to_max_time(arr, max_t), lab))

    return out


def main() -> int:
    args = parse_args()
    zip_dir = args.zip_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(zip_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip files in {zip_dir}")

    all_samples: List[Tuple[str, np.ndarray, int]] = []
    for zp in zips:
        try:
            all_samples.extend(extract_samples(zp, args.max_frame))
        except Exception as e:
            print(f"[WARN] skip {zp.name}: {e}")

    if not all_samples:
        raise RuntimeError("No valid sequences extracted. Check action labels and skeletons.")

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * args.val_ratio))
    if len(all_samples) <= 2:
        n_val = 1
    train_samples = all_samples[:-n_val]
    val_samples = all_samples[-n_val:]

    def pack(samples: List[Tuple[str, np.ndarray, int]]) -> Tuple[np.ndarray, Tuple[List[str], List[int]]]:
        names: List[str] = []
        labels: List[int] = []
        stacks: List[np.ndarray] = []
        for n, arr, lb in samples:
            names.append(n)
            labels.append(lb)
            stacks.append(arr)
        data = np.stack(stacks, axis=0).astype(np.float32)
        return data, (names, labels)

    train_data, train_lab = pack(train_samples)
    val_data, val_lab = pack(val_samples)

    np.save(out_dir / "train_data.npy", train_data)
    np.save(out_dir / "val_data.npy", val_data)
    with (out_dir / "train_label.pkl").open("wb") as f:
        pickle.dump(train_lab, f)
    with (out_dir / "val_label.pkl").open("wb") as f:
        pickle.dump(val_lab, f)

    meta = {
        "classes": list(ACTION_CLASSES),
        "num_class": len(ACTION_CLASSES),
        "max_frame": args.max_frame,
        "n_train": train_data.shape[0],
        "n_val": val_data.shape[0],
        "shape": "(N, 3, T, V, M) with V=17 COCO, M=1",
    }
    print("Saved:", out_dir)
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
