from __future__ import annotations

from collections.abc import Iterable
import csv
from pathlib import Path
import random
import shutil
from typing import Annotated

from loguru import logger
import typer

from ai_detector_model.core.config import PROCESSED_DATA_DIR

app = typer.Typer(add_completion=False)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _collect_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if _is_image(path))


def _read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_images_by_filename(root: Path) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    for image_path in _collect_images(root):
        indexed.setdefault(image_path.name, []).append(image_path)
    for paths in indexed.values():
        paths.sort()
    return indexed


def _ensure_exists(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise typer.BadParameter(f"Missing required paths:\n{joined}")


@app.command()
def main(
    artifact_root: Annotated[
        Path,
        typer.Argument(help="Path to local ArtiFact dataset root directory."),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="Output root, default: data/processed"),
    ] = PROCESSED_DATA_DIR,
    seed: Annotated[
        int,
        typer.Option(help="Seed used for fake data train/test split."),
    ] = 42,
    train_ratio: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Train ratio for fake sources."),
    ] = 0.75,
    clean: Annotated[
        bool,
        typer.Option(
            help="Remove existing data/processed/train and data/processed/test before copying."
        ),
    ] = True,
) -> None:
    """Copy selected ArtiFact sources into train/test folder structure used by loaders."""
    artifact_root = artifact_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    fake_sources = {
        "latent_diffusion": artifact_root / "Fake" / "latent_diffusion",
        "stable_diffusion": artifact_root / "Fake" / "stable_diffusion",
    }
    coco_root = artifact_root / "Real" / "coco"

    _ensure_exists([*fake_sources.values(), coco_root])

    latent_metadata = fake_sources["latent_diffusion"] / "metadata.csv"
    stable_metadata = fake_sources["stable_diffusion"] / "metadata.csv"
    coco_metadata = coco_root / "metadata.csv"

    _ensure_exists([latent_metadata, stable_metadata, coco_metadata])

    train_root = output_root / "train"
    test_root = output_root / "test"
    train_real_root = train_root / "0_real"
    train_fake_root = train_root / "1_fake"
    test_real_root = test_root / "0_real"
    test_fake_root = test_root / "1_fake"

    if clean:
        if train_root.exists():
            shutil.rmtree(train_root)
        if test_root.exists():
            shutil.rmtree(test_root)

    for path in [train_fake_root, test_fake_root, train_real_root, test_real_root]:
        path.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting fake data from metadata.csv")
    fake_entries: list[tuple[str, Path, Path]] = []
    for source_name, source_root in fake_sources.items():
        indexed_images = _index_images_by_filename(source_root)
        rows = _read_metadata(source_root / "metadata.csv")
        for row in rows:
            filename = row["filename"]
            candidates = indexed_images.get(filename, [])
            if not candidates:
                raise FileNotFoundError(f"Could not find {filename} under {source_root}")
            image_path = candidates.pop(0)
            relative_path = image_path.relative_to(source_root)
            fake_entries.append((source_name, image_path, relative_path))
        logger.info(f"Found {len(rows)} files in {source_name}")

    fake_entries.sort(key=lambda item: f"{item[0]}/{item[2]}")
    rng = random.Random(seed)
    rng.shuffle(fake_entries)

    split_idx = int(len(fake_entries) * train_ratio)
    train_fake_entries = fake_entries[:split_idx]
    test_fake_entries = fake_entries[split_idx:]

    train_ids = {f"{source}/{relative}" for source, _, relative in train_fake_entries}
    test_ids = {f"{source}/{relative}" for source, _, relative in test_fake_entries}
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError("Split leakage detected in fake samples")

    fake_train_copied = 0
    fake_test_copied = 0
    for source_name, file_path, relative_path in train_fake_entries:
        destination_file = train_fake_root / source_name / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination_file)
        fake_train_copied += 1

    for source_name, file_path, relative_path in test_fake_entries:
        destination_file = test_fake_root / source_name / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination_file)
        fake_test_copied += 1

    logger.info(
        "Fake split complete: "
        f"train={fake_train_copied}, test={fake_test_copied}, seed={seed}, ratio={train_ratio}"
    )

    logger.info("Copying real/coco from metadata.csv")
    real_train_copied = 0
    real_test_copied = 0
    for row in _read_metadata(coco_metadata):
        image_path = coco_root / row["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Could not find {image_path}")

        category = row["category"]
        if category in {"train2017", "val2017"}:
            destination_file = train_real_root / "coco" / row["image_path"]
            real_train_copied += 1
        elif category == "test2017":
            destination_file = test_real_root / "coco" / row["image_path"]
            real_test_copied += 1
        else:
            raise ValueError(f"Unsupported COCO category: {category}")

        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination_file)

    logger.success(
        "Data prepared successfully. "
        f"train/fake={fake_train_copied}, train/real={real_train_copied}, "
        f"test/fake={fake_test_copied}, test/real={real_test_copied}."
    )


if __name__ == "__main__":
    app()
