from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Annotated

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import torch
import typer

from ai_detector_model.core.config import PYTORCH_MODELS_DIR

app = typer.Typer(add_completion=False)
DEFAULT_OUTPUT_ROOT = PYTORCH_MODELS_DIR


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _resolve_latest_version(client: MlflowClient, model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise typer.BadParameter(f"No versions found for registered model: {model_name}")
    latest = max(versions, key=lambda item: int(item.version))
    return latest.version


@app.command()
def main(
    registered_model_name: str = typer.Argument(..., help="Registered MLflow model name."),
    version: str | None = typer.Option(
        None, "--version", help="Model version. If omitted, latest version is used."
    ),
    output_root: Annotated[
        Path,
        typer.Option(help="Base directory for exported PyTorch models."),
    ] = DEFAULT_OUTPUT_ROOT,
) -> None:
    client = MlflowClient()
    model_version = version or _resolve_latest_version(client, registered_model_name)

    model_uri = f"models:/{registered_model_name}/{model_version}"
    logger.info(f"Loading model from {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)

    model_info = client.get_model_version(registered_model_name, model_version)
    run = client.get_run(model_info.run_id)
    run_data = run.data

    metadata = {
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "model_uri": model_uri,
        "run_id": model_info.run_id,
        "source": model_info.source,
        "status": model_info.status,
        "run_name": run.info.run_name,
        "model_name": run_data.params.get("model.model_name") or run_data.params.get("model_name"),
        "model_type": run_data.params.get("model.model_type") or run_data.params.get("model_type"),
        "input_size": run_data.params.get("model.input_size") or run_data.params.get("input_size"),
        "class_to_idx": run_data.params.get("labels"),
        "device": run_data.params.get("device"),
        "use_amp": run_data.params.get("use_amp"),
        "metrics": run_data.metrics,
    }

    export_dir = output_root / f"{_safe_name(registered_model_name)}_v{_safe_name(model_version)}"
    export_dir.mkdir(parents=True, exist_ok=True)

    pth_path = export_dir / "model.pth"
    metadata_path = export_dir / "metadata.json"

    torch.save(model.state_dict(), pth_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    logger.success(f"Saved model to {pth_path}")
    logger.success(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    app()
