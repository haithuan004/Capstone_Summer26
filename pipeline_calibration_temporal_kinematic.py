from __future__ import annotations

"""
Pipeline 3 bước (theo workflow đề xuất):

  1. Calibration: trên mỗi track, lưu chiều dài tối đa từng đoạn xương
     (cánh tay, cẳng tay, đùi, ống chân) khi cả hai đầu mút đều quan sát được.

  2. Temporal: nội suy tuyến tính theo thời gian giữa các frame có keypoint
     (tái sử dụng logic interpolate_keypoints_in_zips).

  3. Kinematic correction:
     - Tuy chon: noi suy khong gian (mirror / neighbor) cho diem con thieu
       (kinematic_spatial_impute_zips.impute_one_skeleton).
     - Voi tung doan cha-con, neu D > L_max + epsilon thi keo con ve phia cha
       doc vector hien tai sao cho ||con - cha|| = L_max.

Chạy ví dụ:
  py pipeline_calibration_temporal_kinematic.py --input-dir . --output-dir enhanced_zips
  py pipeline_calibration_temporal_kinematic.py --input-dir interpolated_zips --overwrite
"""

import argparse
from math import hypot
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from interpolate_keypoints_in_zips import (
    interpolate_track,
    read_xml_from_zip,
    write_updated_zip,
)
from kinematic_spatial_impute_zips import (
    fmt_xy,
    get_points_by_label,
    impute_one_skeleton,
    is_visible,
    parse_xy,
)

# Parent -> child order: correct from proximal to distal.
ORDERED_LIMB_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

MIN_SEGMENT_PX = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration, temporal interpolation, optional spatial imputation, "
            "and kinematic limb-length correction for CVAT skeleton zips."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Folder containing source .zip files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("enhanced_zips"),
        help="Output folder for processed zips (unless --overwrite).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_enhanced",
        help="Suffix for output names when not using --overwrite on same folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite each zip in --input-dir in place (ignores --output-dir for paths).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=5.0,
        help="Slack in pixels: clamp only if D > L_max + epsilon (default 5).",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Skip spatial imputation (mirror/neighbor) after temporal step.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-zip statistics.",
    )
    return parser.parse_args()


def segment_length(
    skeleton: ET.Element, parent: str, child: str
) -> float | None:
    pb = get_points_by_label(skeleton)
    na = pb.get(parent)
    nb = pb.get(child)
    if na is None or nb is None:
        return None
    if not is_visible(na) or not is_visible(nb):
        return None
    pa = parse_xy(na.get("points"))
    pc = parse_xy(nb.get("points"))
    if pa is None or pc is None:
        return None
    d = hypot(pc[0] - pa[0], pc[1] - pa[1])
    if d < MIN_SEGMENT_PX:
        return None
    return d


def calibrate_track_limb_lengths(track: ET.Element) -> dict[tuple[str, str], float]:
    """Bước 1: L_max cho từng cặp (parent, child) trên toàn bộ frame của track."""
    lengths: dict[tuple[str, str], float] = {}
    for sk in track.findall("./skeleton"):
        for par, chi in ORDERED_LIMB_SEGMENTS:
            d = segment_length(sk, par, chi)
            if d is None:
                continue
            key = (par, chi)
            lengths[key] = max(lengths.get(key, 0.0), d)
    return lengths


def calibrate_skeleton_limb_lengths(skeleton: ET.Element) -> dict[tuple[str, str], float]:
    """Một frame (image-mode): L_max = đoạn đo được trên chính skeleton đó."""
    lengths: dict[tuple[str, str], float] = {}
    for par, chi in ORDERED_LIMB_SEGMENTS:
        d = segment_length(skeleton, par, chi)
        if d is None:
            continue
        lengths[(par, chi)] = d
    return lengths


def apply_limb_length_clamp(
    skeleton: ET.Element,
    limb_lengths: dict[tuple[str, str], float],
    epsilon: float,
) -> int:
    """
    Step 3: If D > L_max + epsilon, place child on ray parent->child at distance L_max.
    """
    points_by_label = get_points_by_label(skeleton)
    coords: dict[str, tuple[float, float]] = {}
    for label, node in points_by_label.items():
        if not is_visible(node):
            continue
        xy = parse_xy(node.get("points"))
        if xy is not None:
            coords[label] = xy

    fixed = 0
    for parent, child in ORDERED_LIMB_SEGMENTS:
        key = (parent, child)
        l_max = limb_lengths.get(key)
        if l_max is None or l_max < MIN_SEGMENT_PX:
            continue
        if parent not in coords or child not in coords:
            continue
        px, py = coords[parent]
        cx, cy = coords[child]
        dx, dy = cx - px, cy - py
        dist = hypot(dx, dy)
        if dist < 1e-6:
            continue
        if dist <= l_max + epsilon:
            continue
        scale = l_max / dist
        nx, ny = px + dx * scale, py + dy * scale
        node_c = points_by_label.get(child)
        if node_c is None:
            continue
        node_c.set("points", fmt_xy(nx, ny))
        coords[child] = (nx, ny)
        fixed += 1
    return fixed


def process_track(
    track: ET.Element,
    epsilon: float,
    use_spatial: bool,
) -> tuple[dict[tuple[str, str], float], int, int, int, int]:
    """
    Returns:
        limb_lengths, temporal_skeletons, temporal_points, spatial_pts, clamp_fixes
    """
    limb_lengths = calibrate_track_limb_lengths(track)
    sk_before = len(track.findall("./skeleton"))
    created, interp_pts = interpolate_track(track)
    sk_after = len(track.findall("./skeleton"))

    spatial_pts = 0
    clamp_fixes = 0
    for sk in track.findall("./skeleton"):
        if use_spatial:
            spatial_pts += impute_one_skeleton(sk)
        clamp_fixes += apply_limb_length_clamp(sk, limb_lengths, epsilon)

    _ = (sk_before, sk_after)  # reserved for verbose extensions
    return limb_lengths, created, interp_pts, spatial_pts, clamp_fixes


def process_image_skeletons(
    root: ET.Element,
    epsilon: float,
    use_spatial: bool,
) -> tuple[int, int, int]:
    """Image-mode: không có track; mỗi skeleton tự calibrate + spatial + clamp."""
    spatial_total = 0
    clamp_total = 0
    n = 0
    for image in root.findall("./image"):
        for sk in image.findall("./skeleton"):
            n += 1
            limb_lengths = calibrate_skeleton_limb_lengths(sk)
            if use_spatial:
                spatial_total += impute_one_skeleton(sk)
            clamp_total += apply_limb_length_clamp(sk, limb_lengths, epsilon)
    return n, spatial_total, clamp_total


def process_zip(
    zip_path: Path,
    out_path: Path,
    epsilon: float,
    use_spatial: bool,
    verbose: bool,
) -> None:
    xml_name, root = read_xml_from_zip(zip_path)

    tracks = root.findall("./track")
    total_created = total_interp = total_spatial = total_clamp = 0
    seg_counts = 0

    if tracks:
        for tr in tracks:
            limb_lengths, created, interp_pts, spatial_pts, clamp_fixes = process_track(
                tr, epsilon=epsilon, use_spatial=use_spatial
            )
            total_created += created
            total_interp += interp_pts
            total_spatial += spatial_pts
            total_clamp += clamp_fixes
            seg_counts += len(limb_lengths)
    else:
        n_sk, spatial_total, clamp_total = process_image_skeletons(
            root, epsilon=epsilon, use_spatial=use_spatial
        )
        total_spatial += spatial_total
        total_clamp += clamp_total
        if verbose:
            print(f"    image-mode skeletons: {n_sk}")

    write_updated_zip(zip_path, out_path, xml_name, root)

    if verbose:
        print(
            f"  temporal: new_frames~{total_created}, interp_points={total_interp} | "
            f"spatial={total_spatial} | limb_clamps={total_clamp} | "
            f"calibrated_segments(tracks)={seg_counts}"
        )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    use_spatial = not args.no_spatial

    zips = sorted(input_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip files in {input_dir}")

    if args.overwrite:
        out_root = input_dir
    else:
        out_root = output_dir
        out_root.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline: calibration -> temporal -> ", end="")
    print("spatial -> " if use_spatial else "(no spatial) -> ", end="")
    print(f"kinematic clamp (epsilon={args.epsilon}px)")
    print(f"Found {len(zips)} zip(s)")

    for z in zips:
        if args.overwrite:
            out_final = out_root / z.name
            tmp = out_final.with_suffix(".tmp.zip")
            process_zip(z, tmp, args.epsilon, use_spatial, args.verbose)
            tmp.replace(out_final)
            print(f"[OK] {z.name} -> {out_final.name} (overwrite)")
        else:
            stem = z.stem.removesuffix("_interp") if z.stem.endswith("_interp") else z.stem
            out_name = f"{stem}{args.suffix}.zip"
            out_path = out_root / out_name
            process_zip(z, out_path, args.epsilon, use_spatial, args.verbose)
            print(f"[OK] {z.name} -> {out_path.name}")

    print(f"Done. Output: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
