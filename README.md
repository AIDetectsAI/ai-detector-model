# ai-detector-model

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python](https://img.shields.io/badge/Python-3.12.12-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![CCDS](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=ONNX&logoColor=white)](https://onnx.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-configuration-5a29e4?logo=hydra)](https://hydra.cc/)
[![OpenCV](https://img.shields.io/badge/OpenCV-CV-blue?logo=opencv)](https://opencv.org/)
[![Pytest](https://img.shields.io/badge/Pytest-fff?logo=pytest&logoColor=000)](https://docs.pytest.org/en/stable/)
[![MkDocs](https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=fff)](#)



A machine learning model designed to classify images as either AI-generated or real, using visual features and patterns to distinguish synthetic content from authentic photographs.

## ⚙️ Project Setup

### Prerequisites
- make
- [uv](https://github.com/astral-sh/uv) package manager | [docs](https://docs.astral.sh/uv/)

### 🕵️ Environment setup

This project uses a `.env` file for configuration. Create it from the template:

```bash
cp .env.example .env
```

### 👟 Quickstart

Install dependencies (uv automatically installs required python version and creates venv) with:

```bash
make requirements
```

Run pre-commit configuration:

```bash
make precommit
```

### 🚀 API

The API uses a locally stored model from the `models` directory.

- The PyTorch model should be stored in `models/pytorch/<MODEL_NAME>/model.pth`
- If the ONNX model is missing, it will be created automatically in `models/onnx/<MODEL_NAME>/model.onnx`
- Set the model directory name in `.env` using `ACTIVE_MODEL_DIR`, for example: `ACTIVE_MODEL_DIR=latent-stable-model`

To start the API:

```bash
make server
```

### 🗂️ Model Storage

Models in this project are stored in two parallel locations:

- `models/pytorch/<MODEL_NAME>/`
- `models/onnx/<MODEL_NAME>/`

The `<MODEL_NAME>` part is the model directory name from `ACTIVE_MODEL_DIR`, for example `latent-stable-model_v1`.

#### PyTorch models

The PyTorch version of a model should be stored in:

```text
models/pytorch/<MODEL_NAME>/model.pth
```

This directory should also contain the MLflow export metadata:

```text
models/pytorch/<MODEL_NAME>/metadata.json
```

The metadata file is used to recover the most important information about the model, such as:

- model architecture name
- model type
- input size
- class mapping
- MLflow run ID
- model version

#### ONNX models

The ONNX export should be stored in:

```text
models/onnx/<MODEL_NAME>/model.onnx
```

This directory should also contain the same metadata file copied from the PyTorch export:

```text
models/onnx/<MODEL_NAME>/metadata.json
```

If the ONNX file is missing, the API will convert the PyTorch model from `models/pytorch/<MODEL_NAME>/` and create the matching ONNX directory automatically.

### 🗄️ Local documentation

You can build and serve the project documentation locally using:

```bash
make build_docs       # build docs
make serve_docs       # serve docs at localhost:9000
```

### ⌨️ Other project commands

Other useful commands include:

- `make clean` – remove compiled Python files and __pycache__ directories
- `make lint` – check code style using ruff
- `make format` – automatically format the code with ruff
- `make test` – run all tests in the tests folder
- `make data` – prepare the dataset using the provided dataset script
- `make server` - start serving model locally

> ℹ️ For more information use: `make help` !

## 📃 Scripts

Repository provides scripts to speed up experimenting and model developing process. Every script is avaliable in `ai_detector_model` subfolder. Detailed description of each script is provided in project documentation [here](https://aidetectsai.github.io/ai-detector-model/experiments/configs/).

### Dataset preparation and evaluation

- `ai_detector_model/utils/build_dataset.py` prepares the processed training dataset from the ArtiFact source tree. It copies real and fake images into the `data/processed/train` and `data/processed/test` folders using the metadata files, and creates a train/test split for fake samples.
- `ai_detector_model/utils/build_final_test_set.py` builds a balanced final test set from the ArtiFact dataset. It selects real and fake sources, removes exact duplicates by file hash, avoids overlap with the existing processed dataset, and writes a manifest plus a JSON report.
- `ai_detector_model/utils/evaluate_final_test.py` evaluates a saved PyTorch checkpoint on the final test set. It loads model metadata, applies the project preprocessing pipeline, computes classification metrics, and can optionally save the report as JSON.

## 🗃️ Documentation

Project documentation is avaliable at adress: https://aidetectsai.github.io/ai-detector-model


## 🌳 Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── configs/
│   │── config.yaml    <- Main configuration entry point
│   │
│   ├── model/         <- Contains model config files
│   ├── data/          <- Dataset & preprocessing config files
│   └── train/         <- Training hyperparameters cofig files
│
├── models
│   ├── onnx           <- ONNX graph models
│   └── pytorch        <- Pytorch models (weights - .ptx + model class - .py)
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         ai_detector_model and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── ai_detector_model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ai_detector_model a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## 👥 Contributions

Currently, external contributions to this project are **not accepted**.

> **Note:** Every pull request **must** be preceded by:

```bash
make format   # Format the code using ruff
make test     # Run all tests to ensure nothing is broken
```
