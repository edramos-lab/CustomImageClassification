import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

class CNNBaseline(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, n)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

class CNNWavelet(CNNBaseline):
    def __init__(self, n):
        super().__init__(n)
        self.wavelet = WaveletTransform(3)
        self.net[0] = nn.Conv2d(12, 32, 3, 1, 1)

    def forward(self, x):
        return super().forward(self.wavelet(x))

class CNNViT(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.cnn = CNNBaseline(256)
        self.patch = nn.Conv2d(256, 128, 4, 4)
        self.cls = nn.Parameter(torch.zeros(1, 1, 128))
        self.pos = nn.Parameter(torch.zeros(1, 17, 128))
        self.tr = nn.Sequential(TransformerEncoder(), TransformerEncoder())
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, n)

    def forward(self, x):
        x = self.cnn.net(x)
        x = self.patch(x).flatten(2).transpose(1, 2)
        B = x.size(0)
        x = torch.cat((self.cls.expand(B, -1, -1), x), 1) + self.pos
        x = self.tr(x)
        return self.fc(self.norm(x[:, 0]))

class WaveletCNNViT(CNNViT):
    def __init__(self, n):
        super().__init__(n)
        self.wavelet = WaveletTransform(3)
        self.cnn.net[0] = nn.Conv2d(12, 32, 3, 1, 1)

    def forward(self, x):
        return super().forward(self.wavelet(x))


MODEL_REGISTRY = {
    "CNNBaseline": CNNBaseline,
    "CNNWavelet": CNNWavelet,
    "CNNViT": CNNViT,
    "WaveletCNNViT": WaveletCNNViT,
}

def get_model_cls(model_cls, model_name):
    if model_cls is not None:
        return model_cls
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {model_name}")
    return MODEL_REGISTRY[model_name]


class ExperimentConfig:
    def __init__(
        self,
        project_name,
        kfolds=5,
        use_stratified=True,
        epochs=20,
        batch_size=16,
        lr=3e-4,
        model_name="CNNBaseline",
        dataset_ratio=1.0
    ):
        self.project_name = project_name
        self.kfolds = kfolds
        self.use_stratified = use_stratified
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_name = model_name
        self.dataset_ratio = dataset_ratio
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

import wandb

def init_wandb(cfg, fold):
    ratio_label = str(cfg.dataset_ratio).replace(".", "p")
    wandb.init(
        project=cfg.project_name,
        name=f"{cfg.model_name}_k{cfg.kfolds}_b{cfg.batch_size}_r{ratio_label}_fold{fold}",
        config={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "kfolds": cfg.kfolds,
            "model": cfg.model_name,
            "dataset_ratio": cfg.dataset_ratio
        },
        reinit=True
    )

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_curve, auc
)
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
def evaluate_test(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)

            y_pred.extend(prob.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
            y_prob.extend(prob.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return metrics, cm, np.array(y_true), np.array(y_prob)
import seaborn as sns
import matplotlib.pyplot as plt

def log_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

from sklearn.preprocessing import label_binarize

def log_roc_curve(y_true, y_prob, class_names):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(7, 7))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    wandb.log({"roc_curve": wandb.Image(fig)})
    plt.close(fig)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        out[:, class_idx].backward()

        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / cam.max()
        return cam

def log_gradcam(model, loader, device, target_layer):
    model.eval()
    cam = GradCAM(model, target_layer)

    x, y = next(iter(loader))
    x = x.to(device)

    heatmap = cam.generate(x[:1], y[0].item()).cpu().numpy()[0]

    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap="jet")
    ax.axis("off")
    wandb.log({"gradcam": wandb.Image(fig)})
    plt.close(fig)

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    losses = []
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    acc = correct / total if total else 0.0
    wandb.log({"train_loss": np.mean(losses), "acc": acc})
    return np.mean(losses)

def _apply_dataset_ratio(trainval_ds, targets, ratio, use_stratified):
    if ratio >= 1.0:
        return trainval_ds, np.asarray(targets)

    indices = np.arange(len(targets))
    if use_stratified:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=ratio,
            random_state=42
        )
        subset_idx, _ = next(splitter.split(indices, targets))
    else:
        rng = np.random.default_rng(42)
        subset_size = max(1, int(len(indices) * ratio))
        subset_idx = rng.choice(indices, size=subset_size, replace=False)

    subset_targets = np.asarray(targets)[subset_idx]
    return Subset(trainval_ds, subset_idx), subset_targets

def run_experiment(model_cls, data_dir, cfg):
    model_cls = get_model_cls(model_cls, cfg.model_name)
    trainval_ds, targets, n_classes = load_dataset(os.path.join(data_dir, "train"))
    trainval_ds, targets = _apply_dataset_ratio(
        trainval_ds,
        targets,
        cfg.dataset_ratio,
        cfg.use_stratified
    )
    test_ds = ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=get_transforms(False)
    )

    splitter = (
        StratifiedKFold(cfg.kfolds, shuffle=True, random_state=42)
        if cfg.use_stratified else
        KFold(cfg.kfolds, shuffle=True, random_state=42)
    )

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(targets)), targets)):
        init_wandb(cfg, fold)

        train_loader = DataLoader(Subset(trainval_ds, tr),
                                  batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(trainval_ds, va),
                                batch_size=cfg.batch_size)
        test_loader = DataLoader(test_ds,
                                 batch_size=cfg.batch_size)

        model = model_cls(n_classes).to(cfg.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(cfg.epochs):
            loss = train_epoch(model, train_loader, opt, loss_fn, cfg.device)
            acc, _ = evaluate(model, val_loader, cfg.device)
            wandb.log({"val_accuracy": acc})

        metrics, cm, y_true, y_prob = evaluate_test(
            model, test_loader, cfg.device
        )

        wandb.log(metrics)
        log_confusion_matrix(cm, test_ds.classes)
        log_roc_curve(y_true, y_prob, test_ds.classes)

        # Grad-CAM (last CNN layer)
        log_gradcam(model, test_loader, cfg.device,
                    target_layer=list(model.modules())[-3])

        wandb.finish()

def _get_param_values(params, name, default):
    spec = params.get(name)
    if spec is None:
        return default
    if "values" in spec:
        return spec["values"]
    if "min" in spec and "max" in spec:
        return [spec["min"], spec["max"]]
    return default

def load_sweep_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    params = cfg.get("parameters", {})
    return params

def run_sweep(model_cls, data_dir, project_name, base_cfg=None, sweep_config_path="conf.yaml"):
    params = load_sweep_config(sweep_config_path)
    kfolds_list = _get_param_values(
        params,
        "kfolds",
        [base_cfg.kfolds] if base_cfg else [5]
    )
    batch_sizes = _get_param_values(
        params,
        "batch_size",
        [base_cfg.batch_size] if base_cfg else [16]
    )
    dataset_ratios = _get_param_values(
        params,
        "dataset_ratio",
        [base_cfg.dataset_ratio] if base_cfg else [1.0]
    )
    model_names = _get_param_values(
        params,
        "model_name",
        [base_cfg.model_name] if base_cfg else ["CNNBaseline"]
    )
    lr_values = _get_param_values(
        params,
        "lr",
        [base_cfg.lr] if base_cfg else [3e-4]
    )
    epochs_values = _get_param_values(
        params,
        "epochs",
        [base_cfg.epochs] if base_cfg else [20]
    )
    stratified_values = _get_param_values(
        params,
        "use_stratified",
        [base_cfg.use_stratified] if base_cfg else [True]
    )

    for model_name in model_names:
        for kfolds in kfolds_list:
            for batch_size in batch_sizes:
                for ratio in dataset_ratios:
                    for lr in lr_values:
                        for epochs in epochs_values:
                            for use_stratified in stratified_values:
                                cfg = ExperimentConfig(
                                    project_name=project_name,
                                    kfolds=kfolds,
                                    use_stratified=use_stratified,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    lr=lr,
                                    model_name=model_name,
                                    dataset_ratio=ratio
                                )
                                run_experiment(model_cls, data_dir, cfg)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run image classification experiments.")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root.")
    parser.add_argument("--project-name", required=True, help="W&B project name.")
    parser.add_argument("--model-name", default="CNNBaseline",
                        choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--dataset-ratio", type=float, default=1.0)
    parser.add_argument("--no-stratified", action="store_true",
                        help="Disable stratified splitting.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep of kfolds/batch/ratio.")
    parser.add_argument("--sweep-config", default="conf.yaml",
                        help="Path to sweep configuration YAML.")
    args = parser.parse_args()

    base_cfg = ExperimentConfig(
        project_name=args.project_name,
        kfolds=args.kfolds,
        use_stratified=not args.no_stratified,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model_name,
        dataset_ratio=args.dataset_ratio
    )

    if args.sweep:
        run_sweep(
            None,
            args.data_dir,
            args.project_name,
            base_cfg=base_cfg,
            sweep_config_path=args.sweep_config
        )
    else:
        run_experiment(None, args.data_dir, base_cfg)
