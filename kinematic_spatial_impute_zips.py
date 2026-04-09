from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple
import xml.etree.ElementTree as ET
import zipfile


# Skeleton topology (COCO-like 17 joints).
JOINT_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "nose": ("left_eye", "right_eye", "left_shoulder", "right_shoulder"),
    "left_eye": ("nose", "left_ear", "right_eye"),
    "right_eye": ("nose", "right_ear", "left_eye"),
    "left_ear": ("left_eye",),
    "right_ear": ("right_eye",),
    "left_shoulder": ("right_shoulder", "left_elbow", "left_hip"),
    "right_shoulder": ("left_shoulder", "right_elbow", "right_hip"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_wrist": ("left_elbow",),
    "right_wrist": ("right_elbow",),
    "left_hip": ("right_hip", "left_shoulder", "left_knee"),
    "right_hip": ("left_hip", "right_shoulder", "right_knee"),
    "left_knee": ("left_hip", "left_ankle"),
    "right_knee": ("right_hip", "right_ankle"),
    "left_ankle": ("left_knee",),
    "right_ankle": ("right_knee",),
}

LEFT_RIGHT_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply kinematic/spatial imputation for missing keypoints in all zip files "
            "inside a folder."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("interpolated_zips"),
        help="Folder containing zip files to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing zip files in-place.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (used when --overwrite is not set).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file details.",
    )
    return parser.parse_args()


def parse_xy(text: str | None) -> Tuple[float, float] | None:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def fmt_xy(x: float, y: float) -> str:
    return f"{x:.2f},{y:.2f}"


def is_visible(point_node: ET.Element) -> bool:
    return point_node.get("outside") != "1"


def get_points_by_label(skeleton_node: ET.Element) -> Dict[str, ET.Element]:
    result: Dict[str, ET.Element] = {}
    for node in skeleton_node.findall("./points"):
        label = node.get("label")
        if label:
            result[label] = node
    return result


def estimate_body_midline_x(coords: Dict[str, Tuple[float, float]]) -> float | None:
    mids: list[float] = []
    for left, right in LEFT_RIGHT_PAIRS:
        if left in coords and right in coords:
            mids.append((coords[left][0] + coords[right][0]) / 2.0)
    if not mids:
        return None
    return sum(mids) / len(mids)


def mirror_from_counterpart(
    label: str, coords: Dict[str, Tuple[float, float]], midline_x: float | None
) -> Tuple[float, float] | None:
    if midline_x is None:
        return None
    for left, right in LEFT_RIGHT_PAIRS:
        if label == left and right in coords:
            xr, yr = coords[right]
            return 2 * midline_x - xr, yr
        if label == right and left in coords:
            xl, yl = coords[left]
            return 2 * midline_x - xl, yl
    return None


def estimate_from_neighbors(
    label: str, coords: Dict[str, Tuple[float, float]]
) -> Tuple[float, float] | None:
    neighbors = JOINT_NEIGHBORS.get(label, ())
    known = [coords[n] for n in neighbors if n in coords]
    if len(known) < 2:
        return None
    x = sum(p[0] for p in known) / len(known)
    y = sum(p[1] for p in known) / len(known)
    return x, y


def enrich_virtual_anchors(coords: Dict[str, Tuple[float, float]]) -> None:
    if "left_shoulder" in coords and "right_shoulder" in coords and "neck" not in coords:
        ls = coords["left_shoulder"]
        rs = coords["right_shoulder"]
        coords["neck"] = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    if "left_hip" in coords and "right_hip" in coords and "mid_hip" not in coords:
        lh = coords["left_hip"]
        rh = coords["right_hip"]
        coords["mid_hip"] = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)


def impute_one_skeleton(skeleton_node: ET.Element) -> int:
    points_by_label = get_points_by_label(skeleton_node)
    coords: Dict[str, Tuple[float, float]] = {}
    missing_labels: list[str] = []

    for label, node in points_by_label.items():
        xy = parse_xy(node.get("points"))
        if xy is not None and is_visible(node):
            coords[label] = xy
        else:
            missing_labels.append(label)

    enrich_virtual_anchors(coords)

    updated = 0
    # Run a few rounds so inferred points can help infer others.
    for _ in range(3):
        if not missing_labels:
            break

        midline_x = estimate_body_midline_x(coords)
        remaining: list[str] = []

        for label in missing_labels:
            estimate = mirror_from_counterpart(label, coords, midline_x)
            if estimate is None:
                estimate = estimate_from_neighbors(label, coords)
            if estimate is None and label in ("nose",) and "neck" in coords:
                estimate = coords["neck"]

            if estimate is None:
                remaining.append(label)
                continue

            node = points_by_label[label]
            node.set("points", fmt_xy(estimate[0], estimate[1]))
            node.set("outside", "0")
            node.set("occluded", "1")
            coords[label] = estimate
            updated += 1

        missing_labels = remaining

    return updated


def update_xml_root(root: ET.Element) -> int:
    total_updates = 0
    for skeleton in root.findall(".//skeleton"):
        total_updates += impute_one_skeleton(skeleton)
    return total_updates


def read_xml_entry(zip_path: Path) -> Tuple[str, ET.Element]:
    with zipfile.ZipFile(zip_path, "r") as src:
        xml_name = next((n for n in src.namelist() if n.lower().endswith(".xml")), None)
        if xml_name is None:
            raise FileNotFoundError(f"No XML file found in {zip_path.name}")
        with src.open(xml_name) as f:
            root = ET.parse(f).getroot()
    return xml_name, root


def write_zip_with_xml(src_zip: Path, dst_zip: Path, xml_name: str, root: ET.Element) -> None:
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(src_zip, "r") as src, zipfile.ZipFile(
        dst_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst:
        for info in src.infolist():
            if info.filename == xml_name:
                dst.writestr(info, xml_bytes)
            else:
                dst.writestr(info, src.read(info.filename))


def process_zip(zip_path: Path, output_path: Path) -> int:
    xml_name, root = read_xml_entry(zip_path)
    updates = update_xml_root(root)
    write_zip_with_xml(zip_path, output_path, xml_name, root)
    return updates


def iter_zip_files(folder: Path) -> Iterable[Path]:
    return sorted(p for p in folder.glob("*.zip") if p.is_file())


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    zip_files = list(iter_zip_files(input_dir))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in: {input_dir}")

    if args.overwrite:
        output_dir = input_dir
    else:
        output_dir = (args.output_dir or (input_dir.parent / f"{input_dir.name}_kinematic")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(zip_files)} zip files")
    total_updates = 0
    for zip_path in zip_files:
        if args.overwrite:
            tmp_out = zip_path.with_suffix(".tmp.zip")
            updates = process_zip(zip_path, tmp_out)
            tmp_out.replace(zip_path)
            out_ref = zip_path.name
        else:
            out_file = output_dir / zip_path.name
            updates = process_zip(zip_path, out_file)
            out_ref = out_file.name
        total_updates += updates

        if args.verbose:
            print(f"[OK] {zip_path.name} -> {out_ref} | imputed_points={updates}")
        else:
            print(f"[OK] {zip_path.name} -> {out_ref}")

    print(f"Done. Total imputed points: {total_updates}")
    print(f"Mode: {'overwrite' if args.overwrite else 'copy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
