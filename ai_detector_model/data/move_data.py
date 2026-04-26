from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
import random
import shutil

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


def _index_images_by_filename(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for image_path in _collect_images(root):
        indexed.setdefault(image_path.name, image_path)
    return indexed


def _ensure_exists(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise typer.BadParameter(f"Missing required paths:\n{joined}")


def _copy_files(files: list[Path], source_root: Path, destination_root: Path) -> int:
    copied = 0
    for file_path in files:
        relative_path = file_path.relative_to(source_root)
        destination_file = destination_root / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination_file)
        copied += 1
    return copied


@app.command()
def main(
    artifact_root: Path = typer.Argument(
        DEFAULT_ARTIFACT_ROOT, help="Path to local ArtiFact dataset root directory."
    ),
    output_root: Path = typer.Option(
        PROCESSED_DATA_DIR, "--output-root", help="Output root, default: data/processed"
    ),
    seed: int = typer.Option(42, help="Seed used for fake data train/test split."),
    train_ratio: float = typer.Option(
        0.75, min=0.0, max=1.0, help="Train ratio for fake sources."
    ),
    clean: bool = typer.Option(
        True,
        help="Remove existing data/processed/train and data/processed/test before copying.",
    ),
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
    train_fake_root = train_root / "fake"
    test_fake_root = test_root / "fake"
    train_real_root = train_root / "real"
    test_real_root = test_root / "real"

    if clean:
        if train_root.exists():
            shutil.rmtree(train_root)
        if test_root.exists():
            shutil.rmtree(test_root)

    for path in [train_fake_root, test_fake_root, train_real_root, test_real_root]:
        path.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting fake data from metadata.csv")
    fake_entries: list[tuple[str, Path]] = []
    for source_name, source_root in fake_sources.items():
        indexed_images = _index_images_by_filename(source_root)
        rows = _read_metadata(source_root / "metadata.csv")
        for row in rows:
            filename = row["filename"]
            image_path = indexed_images.get(filename)
            if image_path is None:
                raise FileNotFoundError(f"Could not find {filename} under {source_root}")
            fake_entries.append((source_name, image_path))
        logger.info(f"Found {len(rows)} files in {source_name}")

    fake_entries.sort(key=lambda item: str(item[1]))
    rng = random.Random(seed)
    rng.shuffle(fake_entries)

    split_idx = int(len(fake_entries) * train_ratio)
    train_fake_entries = fake_entries[:split_idx]
    test_fake_entries = fake_entries[split_idx:]

    fake_train_copied = 0
    fake_test_copied = 0
    for source_name, file_path in train_fake_entries:
        destination_file = train_fake_root / source_name / file_path.name
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination_file)
        fake_train_copied += 1

    for source_name, file_path in test_fake_entries:
        destination_file = test_fake_root / source_name / file_path.name
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
