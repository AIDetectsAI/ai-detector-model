from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Annotated

from loguru import logger
import typer

from ai_detector_model.core.config import PROCESSED_DATA_DIR

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT_ROOT = PROCESSED_DATA_DIR / "final_test"
DEFAULT_EXISTING_ROOT = PROCESSED_DATA_DIR
DEFAULT_EXCLUDE_FAKE = "latent_diffusion,stable_diffusion"
DEFAULT_EXCLUDE_REAL = "coco"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass(frozen=True)
class Sample:
    label_dir: str
    source: str
    source_root: Path
    file_path: Path
    relative_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.label_dir}/{self.source}/{self.relative_path.as_posix()}"


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _parse_csv_list(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _list_sources(parent: Path) -> list[str]:
    if not parent.exists():
        return []
    names: list[str] = []
    for path in sorted(parent.iterdir()):
        if path.is_dir() and (path / "metadata.csv").exists():
            names.append(path.name)
    return names


def _image_index_by_filename(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if _is_image(path):
            index.setdefault(path.name, []).append(path)
    return index


def _resolve_row_path(
    source_root: Path, row: dict[str, str], index: dict[str, list[Path]]
) -> Path | None:
    image_path_value = row.get("image_path", "")
    if image_path_value:
        from_image_path = source_root / image_path_value
        if _is_image(from_image_path):
            return from_image_path

    filename = row.get("filename", "")
    candidates = index.get(filename, [])
    if candidates:
        return candidates.pop(0)
    return None


def _file_md5(path: Path, cache: dict[Path, str]) -> str:
    if path in cache:
        return cache[path]
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    cache[path] = digest
    return digest


def _collect_existing_hashes(existing_root: Path) -> set[str]:
    split_roots = [existing_root / "train", existing_root / "test"]
    hashes: set[str] = set()

    for split_root in split_roots:
        if not split_root.exists():
            continue
        for path in split_root.rglob("*"):
            if _is_image(path):
                hashes.add(hashlib.md5(path.read_bytes()).hexdigest())
    return hashes


def _collect_source_samples(label_dir: str, source_root: Path, source_name: str) -> list[Sample]:
    metadata_path = source_root / "metadata.csv"
    rows = _read_metadata(metadata_path)
    index = _image_index_by_filename(source_root)

    samples: list[Sample] = []
    seen_paths: set[Path] = set()
    for row in rows:
        file_path = _resolve_row_path(source_root, row, index)
        if file_path is None:
            continue
        if file_path in seen_paths:
            continue
        seen_paths.add(file_path)

        relative_path = file_path.relative_to(source_root)
        samples.append(
            Sample(
                label_dir=label_dir,
                source=source_name,
                source_root=source_root,
                file_path=file_path,
                relative_path=relative_path,
            )
        )
    return samples


def _select_sources(
    available: list[str], include_csv: str | None, exclude_csv: str | None
) -> list[str]:
    include = _parse_csv_list(include_csv)
    exclude = _parse_csv_list(exclude_csv)

    available_set = set(available)
    if include:
        missing = include - available_set
        if missing:
            missing_joined = ", ".join(sorted(missing))
            raise typer.BadParameter(f"Requested sources not found: {missing_joined}")
        selected = sorted(include)
    else:
        selected = sorted(available_set - exclude)

    if not selected:
        raise typer.BadParameter("No sources selected after include/exclude filtering")
    return selected


def _shuffle_and_cap(
    grouped_samples: dict[str, list[Sample]], seed: int, max_per_source: int | None
) -> list[Sample]:
    merged: list[Sample] = []
    for source_name in sorted(grouped_samples):
        samples = list(grouped_samples[source_name])
        rng = random.Random(f"{seed}:{source_name}")
        rng.shuffle(samples)
        if max_per_source is not None:
            samples = samples[:max_per_source]
        merged.extend(samples)
    return merged


def _round_robin_select(
    grouped_samples: dict[str, list[Sample]], target_count: int
) -> list[Sample]:
    pools = {source: list(samples) for source, samples in grouped_samples.items()}
    source_names = sorted(pools)
    selected: list[Sample] = []

    while len(selected) < target_count:
        progressed = False
        for source_name in source_names:
            pool = pools[source_name]
            if not pool:
                continue
            selected.append(pool.pop())
            progressed = True
            if len(selected) >= target_count:
                break
        if not progressed:
            break

    return selected


def _prepare_candidates(
    grouped_samples: dict[str, list[Sample]],
    seed: int,
    desired_count: int,
    max_per_source: int | None,
    oversample_ratio: float,
) -> list[Sample]:
    capped: dict[str, list[Sample]] = {}
    for source_name in sorted(grouped_samples):
        samples = list(grouped_samples[source_name])
        rng = random.Random(f"{seed}:{source_name}")
        rng.shuffle(samples)
        if max_per_source is not None:
            samples = samples[:max_per_source]
        capped[source_name] = samples

    prefilter_target = max(desired_count, int(math.ceil(desired_count * oversample_ratio)))
    return _round_robin_select(capped, prefilter_target)


def _deduplicate_and_filter(
    samples: list[Sample],
    forbidden_hashes: set[str],
    hash_cache: dict[Path, str],
) -> tuple[list[Sample], int, int]:
    selected: list[Sample] = []
    seen_hashes: set[str] = set()
    dropped_existing = 0
    dropped_internal = 0

    for sample in samples:
        file_hash = _file_md5(sample.file_path, hash_cache)
        if file_hash in forbidden_hashes:
            dropped_existing += 1
            continue
        if file_hash in seen_hashes:
            dropped_internal += 1
            continue
        seen_hashes.add(file_hash)
        selected.append(sample)

    return selected, dropped_existing, dropped_internal


def _copy_samples(samples: list[Sample], output_root: Path) -> None:
    for sample in samples:
        destination = output_root / sample.label_dir / sample.source / sample.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.file_path, destination)


@app.command()
def main(
    artifact_root: Annotated[Path, typer.Argument(help="Path to local ArtiFact dataset root.")],
    output_root: Annotated[
        Path, typer.Option(help="Output folder for final balanced test set.")
    ] = DEFAULT_OUTPUT_ROOT,
    existing_processed_root: Annotated[
        Path,
        typer.Option(help="Existing processed dataset root to check overlap against."),
    ] = DEFAULT_EXISTING_ROOT,
    fake_sources: Annotated[
        str | None,
        typer.Option(help="Comma-separated fake sources to include (optional)."),
    ] = None,
    real_sources: Annotated[
        str | None,
        typer.Option(help="Comma-separated real sources to include (optional)."),
    ] = None,
    exclude_fake_sources: Annotated[
        str,
        typer.Option(help="Comma-separated fake sources to exclude by default."),
    ] = DEFAULT_EXCLUDE_FAKE,
    exclude_real_sources: Annotated[
        str,
        typer.Option(help="Comma-separated real sources to exclude by default."),
    ] = DEFAULT_EXCLUDE_REAL,
    max_per_source: Annotated[
        int | None,
        typer.Option(min=1, help="Optional cap of samples per source before balancing."),
    ] = None,
    target_total: Annotated[
        int,
        typer.Option(min=2, help="Approximate total number of images in final balanced set."),
    ] = 200000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    clean: Annotated[bool, typer.Option(help="Delete output_root before building.")] = True,
    hash_overlap_check: Annotated[
        bool,
        typer.Option(help="Filter out samples that overlap by exact hash with train/test."),
    ] = True,
) -> None:
    artifact_root = artifact_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    existing_processed_root = existing_processed_root.expanduser().resolve()

    fake_parent = artifact_root / "Fake"
    real_parent = artifact_root / "Real"
    if not fake_parent.exists() or not real_parent.exists():
        raise typer.BadParameter("ArtiFact root must contain Fake/ and Real/ directories")

    available_fake = _list_sources(fake_parent)
    available_real = _list_sources(real_parent)

    selected_fake = _select_sources(available_fake, fake_sources, exclude_fake_sources)
    selected_real = _select_sources(available_real, real_sources, exclude_real_sources)

    logger.info(f"Selected fake sources: {selected_fake}")
    logger.info(f"Selected real sources: {selected_real}")

    fake_grouped: dict[str, list[Sample]] = {}
    real_grouped: dict[str, list[Sample]] = {}

    for source_name in selected_fake:
        source_root = fake_parent / source_name
        samples = _collect_source_samples("1_fake", source_root, source_name)
        if not samples:
            raise RuntimeError(f"No fake samples resolved for source: {source_name}")
        fake_grouped[source_name] = samples
        logger.info(f"Collected fake samples: {source_name} -> {len(samples)}")

    for source_name in selected_real:
        source_root = real_parent / source_name
        samples = _collect_source_samples("0_real", source_root, source_name)
        if not samples:
            raise RuntimeError(f"No real samples resolved for source: {source_name}")
        real_grouped[source_name] = samples
        logger.info(f"Collected real samples: {source_name} -> {len(samples)}")

    target_per_class = target_total // 2
    fake_candidates = _prepare_candidates(
        fake_grouped,
        seed=seed,
        desired_count=target_per_class,
        max_per_source=max_per_source,
        oversample_ratio=1.3,
    )
    real_candidates = _prepare_candidates(
        real_grouped,
        seed=seed,
        desired_count=target_per_class,
        max_per_source=max_per_source,
        oversample_ratio=1.3,
    )

    existing_hashes: set[str] = set()
    if hash_overlap_check:
        logger.info("Building existing hash index from train/test")
        existing_hashes = _collect_existing_hashes(existing_processed_root)
        logger.info(f"Existing hash count: {len(existing_hashes)}")

    hash_cache: dict[Path, str] = {}
    fake_filtered, fake_dropped_existing, fake_dropped_internal = _deduplicate_and_filter(
        fake_candidates, existing_hashes, hash_cache
    )
    real_filtered, real_dropped_existing, real_dropped_internal = _deduplicate_and_filter(
        real_candidates, existing_hashes, hash_cache
    )

    balance_size = min(len(fake_filtered), len(real_filtered), target_per_class)
    if balance_size == 0:
        raise RuntimeError("No samples left after filtering; cannot build balanced final test")

    rng = random.Random(seed)
    rng.shuffle(fake_filtered)
    rng.shuffle(real_filtered)

    final_fake = fake_filtered[:balance_size]
    final_real = real_filtered[:balance_size]
    final_samples = final_real + final_fake

    ids = [sample.sample_id for sample in final_samples]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample_id detected in final selection")

    if clean and output_root.exists():
        shutil.rmtree(output_root)

    (output_root / "0_real").mkdir(parents=True, exist_ok=True)
    (output_root / "1_fake").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    _copy_samples(final_samples, output_root)

    manifest_path = output_root / "manifests" / "final_test_manifest.csv"
    report_path = output_root / "manifests" / "final_test_report.json"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "label",
                "source",
                "original_path",
                "relative_path",
                "md5",
            ],
        )
        writer.writeheader()
        for sample in final_samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label_dir,
                    "source": sample.source,
                    "original_path": str(sample.file_path),
                    "relative_path": sample.relative_path.as_posix(),
                    "md5": _file_md5(sample.file_path, hash_cache),
                }
            )

    report = {
        "mode": "balanced",
        "artifact_root": str(artifact_root),
        "output_root": str(output_root),
        "existing_processed_root": str(existing_processed_root),
        "selected_fake_sources": selected_fake,
        "selected_real_sources": selected_real,
        "max_per_source": max_per_source,
        "target_total": target_total,
        "target_per_class": target_per_class,
        "seed": seed,
        "hash_overlap_check": hash_overlap_check,
        "candidate_counts": {
            "fake": len(fake_candidates),
            "real": len(real_candidates),
        },
        "dropped_counts": {
            "fake_existing_hash_overlap": fake_dropped_existing,
            "real_existing_hash_overlap": real_dropped_existing,
            "fake_internal_hash_duplicates": fake_dropped_internal,
            "real_internal_hash_duplicates": real_dropped_internal,
        },
        "final_counts": {
            "0_real": len(final_real),
            "1_fake": len(final_fake),
            "total": len(final_samples),
        },
        "manifest_path": str(manifest_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    logger.success(
        "Final balanced test set created: "
        f"0_real={len(final_real)}, 1_fake={len(final_fake)}, total={len(final_samples)}"
    )
    logger.success(f"Manifest saved to {manifest_path}")
    logger.success(f"Report saved to {report_path}")


if __name__ == "__main__":
    app()
