from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

TARGET_CLASSES = ("standing", "walking", "sitting", "falling")
CLASS_COLORS = {
    "standing": "#4C78A8",
    "walking": "#F58518",
    "sitting": "#54A24B",
    "falling": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot total class distribution across all zip files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing zip files. Defaults to current directory.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("class_distribution_total.png"),
        help="Path to output chart image.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI used when saving the output image.",
    )
    return parser.parse_args()


def normalize_action(value: str | None) -> str | None:
    if value is None:
        return None
    action = value.strip().lower()
    if action in TARGET_CLASSES:
        return action
    return None


def collect_action_from_attributes(attributes: list[ET.Element]) -> str | None:
    for attribute in attributes:
        if attribute.get("name") != "action":
            continue
        return normalize_action(attribute.text)
    return None


def update_counts_from_xml(root: ET.Element, counts: dict[str, int]) -> None:
    tracks = root.findall("./track")
    images = root.findall("./image")

    if tracks:
        for track in tracks:
            for skeleton in track.findall("./skeleton"):
                action = collect_action_from_attributes(skeleton.findall("./attribute"))
                if action is not None:
                    counts[action] += 1
        return

    for image in images:
        for skeleton in image.findall("./skeleton"):
            action = collect_action_from_attributes(skeleton.findall("./attribute"))
            if action is not None:
                counts[action] += 1


def collect_total_counts(input_dir: Path) -> tuple[dict[str, int], int]:
    counts = {class_name: 0 for class_name in TARGET_CLASSES}
    zip_files = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {input_dir}")

    processed = 0
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path) as archive:
            xml_name = next(
                (name for name in archive.namelist() if name.lower().endswith(".xml")),
                None,
            )
            if xml_name is None:
                continue

            with archive.open(xml_name) as xml_file:
                root = ET.parse(xml_file).getroot()
            update_counts_from_xml(root, counts)
            processed += 1

    if processed == 0:
        raise RuntimeError("No valid XML annotation files found in zip archives.")

    return counts, processed


def plot_total_distribution(counts: dict[str, int], output_image: Path, dpi: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to create the plot. Install it with: pip install matplotlib"
        ) from exc

    labels = list(TARGET_CLASSES)
    values = [counts[class_name] for class_name in labels]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=[CLASS_COLORS[name] for name in labels], width=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(1, total * 0.005),
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Total class distribution (all zip files)")
    ax.set_ylabel("Annotated frame instances")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.suptitle(f"Total annotations: {total}", fontsize=12)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_image, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_image = args.output_image.resolve()

    counts, processed = collect_total_counts(input_dir)
    plot_total_distribution(counts, output_image, args.dpi)

    print(f"Processed {processed} zip files")
    print("Class counts:")
    for class_name in TARGET_CLASSES:
        print(f"  {class_name}: {counts[class_name]}")
    print(f"Saved chart: {output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
