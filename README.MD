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

This runs kfolds {2,5}, batch sizes {32,64}, and dataset ratios
{0.1, 0.25, 0.5, 1.0}.

```
python run-experiment.py \
  --data-dir /path/to/data \
  --project-name my_wandb_project \
  --model-name CNNBaseline \
  --sweep
```

## Available models

- `CNNBaseline`
- `CNNWavelet`
- `CNNViT`
- `WaveletCNNViT`
