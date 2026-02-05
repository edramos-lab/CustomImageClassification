# Custom Image Classification

Minimal training and evaluation script for image classification with
KFold/StratifiedKFold, W&B logging, and optional dataset subset ratios.

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- wandb

## Dataset layout

```
data/
  train/
    class_a/
    class_b/
  test/
    class_a/
    class_b/
```

## Run a single experiment

```
python run-experiment.py \
  --data-dir /path/to/data \
  --project-name my_wandb_project \
  --model-name CNNBaseline \
  --epochs 20 \
  --batch-size 32 \
  --lr 3e-4 \
  --kfolds 5 \
  --dataset-ratio 1.0
```

## Run the full sweep

The sweep reads from `conf.yaml` (including model names). Update the values
there to control the grid.

```
python run-experiment.py \
  --data-dir /path/to/data \
  --project-name my_wandb_project \
  --model-name CNNBaseline \
  --sweep
```

## W&B sweep example

If you want W&B to orchestrate runs, use a W&B sweep config that specifies
the program and arguments. Example `wandb_sweep.yaml`:

```
program: run-experiment.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  lr:
    min: 1e-4
    max: 1e-3
  batch_size:
    values: [32, 64]
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - --data-dir
  - /path/to/data
  - --project-name
  - my_wandb_project
  - --model-name
  - CNNBaseline
```

Run the sweep:

```
wandb login
wandb sweep wandb_sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

Use a different config file:

```
python run-experiment.py \
  --data-dir /path/to/data \
  --project-name my_wandb_project \
  --model-name CNNBaseline \
  --sweep \
  --sweep-config /path/to/conf.yaml
```

## Available models

- `CNNBaseline`
- `CNNWavelet`
- `CNNViT`
- `WaveletCNNViT`
