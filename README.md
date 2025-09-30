# ai-detector-model

<div style="display: flex; flex-wrap: wrap; gap: 5px; align-items: center;">
    <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"> <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /> </a> <!-- Python --> <a target="_blank" href="https://www.python.org/"> <img src="https://img.shields.io/badge/Python-3.11.9-blue?logo=python" /> </a> <!-- PyTorch --> <a target="_blank" href="https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-modeling-red?logo=pytorch" /> </a> <!-- ONNX --> <a target="_blank" href="https://onnx.ai/"> <img src="https://img.shields.io/badge/ONNX-ML-orange?logo=onnx" /> </a> <!-- Hydra --> <a target="_blank" href="https://hydra.cc/"> <img src="https://img.shields.io/badge/Hydra-configuration-5a29e4?logo=hydra" /> </a> <!-- OpenCV --> <a target="_blank" href="https://opencv.org/"> <img src="https://img.shields.io/badge/OpenCV-CV-blue?logo=opencv" /> </a> <!-- pytest --> <a target="_blank" href="https://docs.pytest.org/"> <img src="https://img.shields.io/badge/pytest-testing-4b8bbe?logo=pytest" /> </a> <!-- MkDocs --> <a target="_blank" href="https://www.mkdocs.org/"> <img src="https://img.shields.io/badge/MkDocs-docs-ff69b4?logo=mkdocs" /> </a>
</div>


A machine learning model designed to classify images as either AI-generated or real, using visual features and patterns to distinguish synthetic content from authentic photographs.

## ⚙️ Project Setup

### 👟 Quickstart

To set up the project environment, install dependencies, and prepare for development, you can use the provided `Makefile` commands. First, create a Python environment with the required version:

```bash
make create_environment
```
Next, install all Python dependencies:

```bash
make requirements # For global install

pipenv run make requirements # For pipenv environment install
```

### 🕵️ Environment setup (Optional)

This project uses a `.env` file for configuration. Create it from the template:

```bash
cp .env.example .env
```

### 🗄️ Local documentation

You can build and serve the project documentation locally using:

```bash
make build_docs       # build docs
make serve_docs       # serve docs at localhost:9000
```

### ⌨️ Other project commands

Other useful commands include:

- `make clean` – remove compiled Python files and __pycache__ directories
- `make lint` – check code style using flake8, isort, and black
- `make format` – automatically format the code
- `make test` – run all tests in the tests folder
- `make data` – prepare the dataset using the provided dataset script
- `make server` - start serving model locally

> ℹ️ For more information use: `make help` !

## 📃 Scripts

Repository provides scripts to speed up experimenting and model developing process. Every script is avaliable in `ai_detector_model` subfolder. Detailed description of each script is provided in project documentation [here](https://aidetectsai.github.io/ai-detector-model/experiments/configs/).

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
│                         ai_detector_model and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
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
make format   # Format the code using isort & black
make test     # Run all tests to ensure nothing is broken
```