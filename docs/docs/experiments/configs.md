# Configuring a Training Script with Hydra

This guide explains how to set up and organize a training pipeline using **[Hydra](https://hydra.cc/)**. Hydra enables you to manage hierarchical configurations, making experiments easier to reproduce, customize, and extend.

---

## 📂 Project Structure

The configuration files are organized as follows:

```
configs/
│── config.yaml # Main entry point
│
├── model/
│    └── baseline_model.yaml # Model parameters
│
├── data/
│    └── default.yaml # Dataset & preprocessing
│
└── train/
     └── default.yaml # Training hyperparameters
```


---

## ⚙️ Main Configuration (`configs/config.yaml`)

The **main config** defines defaults, project details, paths, Hydra output behavior, and logging:

```yaml
defaults:
  - model: baseline_model
  - data: default
  - train: default
  - _self_

project:
  name: ai-detector-model
  seed: 42

paths:
  project_root: ..
  data: ${paths.project_root}/data
  models: ${paths.project_root}/models
  outputs: ${paths.project_root}/reports/experiments

hydra:
  run:
    dir: ${paths.outputs}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: ${paths.outputs}/multirun/${now:%Y-%m-%d_%H-%M-%S}

logging:
  level: INFO
```

## 🧠 Model Configuration (`configs/model/`)

This folders contains configuration yaml files for models structure and parameters.
Example configuration file for baseline model (`baseline_model.yaml`):

```yaml
model:
  name: custom_binary_cnn
  input_channels: 3
  num_classes: 1
  channels: [32, 64, 128, 256]
  stage_dropout: 0.2
  global_pool: true
  fc_hidden: 128
  fc_dropout: 0.5
  activation: relu
  final_activation: sigmoid
```

## 📊 Data Configuration (`configs/data/`)

The dataset config includes paths, dataloader settings, and transforms.
Configuration file for project data (`default.yaml`):

```yaml
dataset:
  name: processed_dataset
  path: ${paths.data}/processed/
  channels: 3
  batch_size: 64
  shuffle: true
  num_workers: 4
  pin_memory: true

  transforms:
    resize: [148, 148]
    crop_size: [128, 128]
    horizontal_flip: true
    vertical_flip: false
    rotation: 15
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

  split:
    train: 0.7
    val: 0.15
    test: 0.15
```

## 🚀 Training Configuration (`configs/train/`)

Training configuration allows for hyperparameter specialisation.
Project train hyperparameters:

```yaml
optimizer:
  type: adam
  lr: 0.001
  weight_decay: 1e-4

scheduler:
  type: step_lr
  step_size: 10
  gamma: 0.1

epochs: 30
grad_clip: 1.0
log_interval: 10
checkpoint_interval: 5
```

## ▶️ Running Experiments

To run with the default config:

```bash
py train.py
```

To override specific parameters from the command line:

```bash
python train.py train.epochs=50 optimizer.lr=0.0005
```

To run a sweep (grid search/random search):

```bash
python train.py -m train.epochs=20,30 optimizer.lr=0.001,0.0001
```

This will create multiple runs under the multirun directory with different combinations.
