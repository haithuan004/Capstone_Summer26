from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate keypoints along time axis for each track in each zip annotation file."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing source zip files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("interpolated_zips"),
        help="Directory to save interpolated zip files.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_interp",
        help="Suffix appended to each output zip name.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file processing details.",
    )
    return parser.parse_args()


def parse_xy(value: str | None) -> Tuple[float, float] | None:
    if not value:
        return None
    parts = [item.strip() for item in value.split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def format_xy(x: float, y: float) -> str:
    return f"{x:.2f},{y:.2f}"


def is_point_visible(point_node: ET.Element) -> bool:
    return point_node.get("outside") != "1"


def iter_track_skeletons(track: ET.Element) -> List[Tuple[int, ET.Element]]:
    rows: List[Tuple[int, ET.Element]] = []
    for skeleton in track.findall("./skeleton"):
        frame_text = skeleton.get("frame")
        if frame_text is None:
            continue
        try:
            frame_id = int(frame_text)
        except ValueError:
            continue
        rows.append((frame_id, skeleton))
    rows.sort(key=lambda item: item[0])
    return rows


def gather_labels(track: ET.Element) -> List[str]:
    labels: set[str] = set()
    for skeleton in track.findall("./skeleton"):
        for point in skeleton.findall("./points"):
            label = point.get("label")
            if label:
                labels.add(label)
    return sorted(labels)


def find_point_node(skeleton: ET.Element, label: str) -> ET.Element | None:
    for point in skeleton.findall("./points"):
        if point.get("label") == label:
            return point
    return None


def get_or_create_skeleton(
    track: ET.Element,
    frame_to_skeleton: Dict[int, ET.Element],
    sorted_frames: List[int],
    target_frame: int,
) -> ET.Element:
    existing = frame_to_skeleton.get(target_frame)
    if existing is not None:
        return existing

    insert_index = 0
    while insert_index < len(sorted_frames) and sorted_frames[insert_index] < target_frame:
        insert_index += 1

    if insert_index > 0:
        template = frame_to_skeleton[sorted_frames[insert_index - 1]]
    else:
        template = frame_to_skeleton[sorted_frames[insert_index]]

    skeleton = ET.Element("skeleton")
    skeleton.set("frame", str(target_frame))
    skeleton.set("keyframe", "0")
    skeleton.set("z_order", template.get("z_order", "0"))

    # Carry over attributes (e.g. action, is_crowd) from nearest template.
    for attr in template.findall("./attribute"):
        skeleton.append(copy.deepcopy(attr))

    track.append(skeleton)
    frame_to_skeleton[target_frame] = skeleton
    sorted_frames.append(target_frame)
    sorted_frames.sort()
    reorder_skeletons(track)
    return skeleton


def get_or_create_point_node(skeleton: ET.Element, label: str) -> ET.Element:
    point = find_point_node(skeleton, label)
    if point is not None:
        return point
    point = ET.Element("points")
    point.set("label", label)
    point.set("keyframe", skeleton.get("keyframe", "0"))
    point.set("outside", "0")
    point.set("occluded", "0")
    point.set("points", "0.00,0.00")
    skeleton.append(point)
    return point


def reorder_skeletons(track: ET.Element) -> None:
    nodes = list(track)
    skeleton_nodes = [n for n in nodes if n.tag == "skeleton"]
    other_nodes = [n for n in nodes if n.tag != "skeleton"]
    skeleton_nodes.sort(key=lambda n: int(n.get("frame", "0")))

    for node in list(track):
        track.remove(node)
    for node in skeleton_nodes:
        track.append(node)
    for node in other_nodes:
        track.append(node)


def interpolate_track(track: ET.Element) -> Tuple[int, int]:
    skeleton_rows = iter_track_skeletons(track)
    if len(skeleton_rows) < 2:
        return 0, 0

    frame_to_skeleton = {frame: sk for frame, sk in skeleton_rows}
    sorted_frames = [frame for frame, _ in skeleton_rows]
    labels = gather_labels(track)

    created_skeletons: set[int] = set()
    updated_points = 0

    for label in labels:
        observations: List[Tuple[int, float, float]] = []
        for frame_id in sorted_frames:
            skeleton = frame_to_skeleton[frame_id]
            point = find_point_node(skeleton, label)
            if point is None or not is_point_visible(point):
                continue
            xy = parse_xy(point.get("points"))
            if xy is None:
                continue
            observations.append((frame_id, xy[0], xy[1]))

        if len(observations) < 2:
            continue

        for (f1, x1, y1), (f2, x2, y2) in zip(observations, observations[1:]):
            if f2 - f1 <= 1:
                continue
            gap = f2 - f1
            for frame_id in range(f1 + 1, f2):
                alpha = (frame_id - f1) / gap
                xi = x1 + alpha * (x2 - x1)
                yi = y1 + alpha * (y2 - y1)

                skeleton = get_or_create_skeleton(track, frame_to_skeleton, sorted_frames, frame_id)
                if frame_id not in created_skeletons and frame_id not in [f for f, _ in skeleton_rows]:
                    created_skeletons.add(frame_id)

                point = get_or_create_point_node(skeleton, label)
                point.set("points", format_xy(xi, yi))
                point.set("outside", "0")
                point.set("occluded", point.get("occluded", "0"))
                point.set("keyframe", skeleton.get("keyframe", "0"))
                updated_points += 1

    return len(created_skeletons), updated_points


def read_xml_from_zip(zip_path: Path) -> Tuple[str, ET.Element]:
    with zipfile.ZipFile(zip_path, "r") as src_zip:
        xml_name = next((n for n in src_zip.namelist() if n.lower().endswith(".xml")), None)
        if xml_name is None:
            raise FileNotFoundError(f"No XML annotation found in {zip_path.name}")
        with src_zip.open(xml_name) as xml_file:
            root = ET.parse(xml_file).getroot()
    return xml_name, root


def write_updated_zip(
    src_zip_path: Path,
    dst_zip_path: Path,
    xml_name: str,
    root: ET.Element,
) -> None:
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(src_zip_path, "r") as src_zip, zipfile.ZipFile(
        dst_zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst_zip:
        for info in src_zip.infolist():
            if info.filename == xml_name:
                dst_zip.writestr(info, xml_bytes)
            else:
                dst_zip.writestr(info, src_zip.read(info.filename))


def process_zip(zip_path: Path, output_dir: Path, suffix: str) -> Tuple[Path, int, int]:
    xml_name, root = read_xml_from_zip(zip_path)
    total_created = 0
    total_updated = 0

    for track in root.findall("./track"):
        created, updated = interpolate_track(track)
        total_created += created
        total_updated += updated

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{zip_path.stem}{suffix}.zip"
    out_path = output_dir / out_name
    write_updated_zip(zip_path, out_path, xml_name, root)
    return out_path, total_created, total_updated


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    zip_files = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {input_dir}")

    print(f"Found {len(zip_files)} zip files")
    for zip_path in zip_files:
        try:
            out_path, created, updated = process_zip(zip_path, output_dir, args.suffix)
            if args.verbose:
                print(
                    f"[OK] {zip_path.name} -> {out_path.name} | "
                    f"new_skeleton_frames={created}, interpolated_points={updated}"
                )
            else:
                print(f"[OK] {zip_path.name} -> {out_path.name}")
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Skip {zip_path.name}: {exc}")

    print(f"Done. Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
