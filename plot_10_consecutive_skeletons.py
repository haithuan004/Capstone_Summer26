from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET


# COCO-style skeleton connections for 17 keypoints.
SKELETON_EDGES: List[Tuple[str, str]] = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw N random consecutive skeleton frames from one or more zip annotations. "
            "Use --from-folder or --preset to pick Data_origin vs interpolated vs enhanced; "
            "otherwise the first matching zip among several folders is used. "
            "Which window of frames is drawn is random unless you set --seed."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Project root used with --preset (default: current directory).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of consecutive frames to draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible frame selection.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for output images.",
    )
    parser.add_argument(
        "--stems",
        type=str,
        default="fall_cafe_01,Action_02",
        help=(
            "Comma-separated base names (no .zip), e.g. fall_cafe_01,Action_02. "
            "Each is resolved as name.zip, name_enhanced.zip, or name_interp.zip "
            "(first existing file wins inside the chosen folder(s))."
        ),
    )
    parser.add_argument(
        "--from-folder",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Search for zips only in this folder (e.g. Data_origin or interpolated_zips). "
            "Overrides --preset and the default multi-folder search."
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=("origin", "interpolated", "enhanced"),
        default=None,
        help=(
            "Shortcut: look only under input-dir/Data_origin, interpolated_zips, or enhanced_zips. "
            "Ignored if --from-folder is set."
        ),
    )
    return parser.parse_args()


def _candidate_zip_names(stem: str) -> List[str]:
    stem = stem.strip()
    if not stem:
        return []
    if stem.lower().endswith(".zip"):
        return [stem]
    return [
        f"{stem}.zip",
        f"{stem}_enhanced.zip",
        f"{stem}_interp.zip",
    ]


def build_search_roots(
    input_dir: Path,
    from_folder: Path | None,
    preset: str | None,
) -> List[Path]:
    """Folders to search, in order. Single folder if --from-folder or --preset."""
    input_dir = input_dir.resolve()
    if from_folder is not None:
        p = from_folder.expanduser().resolve()
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {p}")
        return [p]
    if preset == "origin":
        return [input_dir / "Data_origin"]
    if preset == "interpolated":
        return [input_dir / "interpolated_zips"]
    if preset == "enhanced":
        return [input_dir / "enhanced_zips"]
    return [
        input_dir,
        input_dir / "Data_origin",
        input_dir / "interpolated_zips",
        input_dir / "enhanced_zips",
        input_dir / "_test_enhanced",
    ]


def resolve_zip_path(roots: Sequence[Path], stem: str) -> Path:
    """First matching zip under roots (in order), for each candidate filename."""
    for name in _candidate_zip_names(stem):
        for root in roots:
            candidate = (root / name).resolve()
            if candidate.is_file():
                return candidate
    tried = ", ".join(_candidate_zip_names(stem))
    existing = [str(r) for r in roots if r.exists()]
    raise FileNotFoundError(
        f"Could not find zip for stem {stem!r}. Tried [{tried}] under: {existing}"
    )


def parse_xy(points_text: str | None) -> Tuple[float, float] | None:
    if not points_text:
        return None
    parts = [item.strip() for item in points_text.split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def read_xml_root(zip_path: Path) -> ET.Element:
    with zipfile.ZipFile(zip_path) as archive:
        xml_name = next((n for n in archive.namelist() if n.lower().endswith(".xml")), None)
        if xml_name is None:
            raise FileNotFoundError(f"No XML file found in {zip_path.name}")
        with archive.open(xml_name) as xml_file:
            return ET.parse(xml_file).getroot()


def skeleton_points_map(skeleton_node: ET.Element) -> Dict[str, Tuple[float, float]]:
    points_map: Dict[str, Tuple[float, float]] = {}
    for point_node in skeleton_node.findall("./points"):
        label = point_node.get("label")
        xy = parse_xy(point_node.get("points"))
        outside = point_node.get("outside")
        if not label or xy is None:
            continue
        # Skip invisible keypoints.
        if outside == "1":
            continue
        points_map[label] = xy
    return points_map


def collect_track_skeletons(root: ET.Element) -> List[Tuple[int, Dict[str, Tuple[float, float]]]]:
    per_frame: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for track in root.findall("./track"):
        for skeleton in track.findall("./skeleton"):
            frame_text = skeleton.get("frame")
            if frame_text is None:
                continue
            try:
                frame_id = int(frame_text)
            except ValueError:
                continue
            points_map = skeleton_points_map(skeleton)
            if not points_map:
                continue
            # If multiple persons appear in same frame, keep the richest skeleton.
            previous = per_frame.get(frame_id)
            if previous is None or len(points_map) > len(previous):
                per_frame[frame_id] = points_map

    return sorted(per_frame.items(), key=lambda item: item[0])


def pick_random_consecutive_frames(
    frame_items: List[Tuple[int, Dict[str, Tuple[float, float]]]], num_frames: int
) -> List[Tuple[int, Dict[str, Tuple[float, float]]]]:
    if not frame_items:
        raise ValueError("No valid skeleton frames found.")

    frame_ids = [item[0] for item in frame_items]
    frame_map = {fid: points for fid, points in frame_items}

    valid_starts: List[int] = []
    for start in frame_ids:
        window = list(range(start, start + num_frames))
        if all(fid in frame_map for fid in window):
            valid_starts.append(start)

    if not valid_starts:
        raise ValueError(f"No {num_frames} consecutive frames found in this file.")

    chosen_start = random.choice(valid_starts)
    return [(fid, frame_map[fid]) for fid in range(chosen_start, chosen_start + num_frames)]


def draw_sequence(
    sequence: List[Tuple[int, Dict[str, Tuple[float, float]]]],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 5
    rows = (len(sequence) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), constrained_layout=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()  # image-like coordinate system
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    for idx, (frame_id, points_map) in enumerate(sequence):
        ax = axes[idx]

        # Draw bones first.
        for p1, p2 in SKELETON_EDGES:
            if p1 in points_map and p2 in points_map:
                x1, y1 = points_map[p1]
                x2, y2 = points_map[p2]
                ax.plot([x1, x2], [y1, y2], color="#1f77b4", linewidth=1.8)

        # Draw joints.
        xs = [xy[0] for xy in points_map.values()]
        ys = [xy[1] for xy in points_map.values()]
        ax.scatter(xs, ys, c="#d62728", s=20)
        ax.set_title(f"frame {frame_id}", fontsize=9)

        # Tight local limits for readability.
        if xs and ys:
            pad_x = max(8.0, (max(xs) - min(xs)) * 0.15)
            pad_y = max(8.0, (max(ys) - min(ys)) * 0.15)
            ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
            ax.set_ylim(max(ys) + pad_y, min(ys) - pad_y)

    for idx in range(len(sequence), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_one_zip(zip_path: Path, num_frames: int, dpi: int) -> Path:
    root = read_xml_root(zip_path)
    frame_items = collect_track_skeletons(root)
    sequence = pick_random_consecutive_frames(frame_items, num_frames)

    output_path = zip_path.with_name(f"{zip_path.stem}_10_consecutive_skeletons.png")
    draw_sequence(
        sequence=sequence,
        title=f"{zip_path.stem}: {num_frames} random consecutive skeleton frames",
        output_path=output_path,
        dpi=dpi,
    )
    return output_path


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    input_dir = args.input_dir.resolve()
    roots = build_search_roots(input_dir, args.from_folder, args.preset)
    stems = [s.strip() for s in args.stems.split(",") if s.strip()]

    for stem in stems:
        zip_path = resolve_zip_path(roots, stem)
        output = process_one_zip(zip_path, num_frames=args.num_frames, dpi=args.dpi)
        print(f"Saved: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
