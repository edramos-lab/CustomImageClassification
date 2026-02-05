import os
import tempfile
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class WaveletTransform(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch

    def forward(self, x):
        # Simple 4-band expansion to match expected channel count.
        return torch.cat([x, x, x, x], dim=1)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

def get_transforms(is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def load_dataset(train_dir):
    ds = ImageFolder(root=train_dir, transform=get_transforms(True))
    targets = [label for _, label in ds.samples]
    n_classes = len(ds.classes)
    return ds, targets, n_classes

def evaluate(model, loader, device, amp=False):
    model.eval()
    correct = 0
    total = 0
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ctx = (
                torch.amp.autocast(device_type="cuda")
                if amp and device == "cuda"
                else nullcontext()
            )
            with ctx:
                logits = model(x)
            preds = logits.argmax(1)
            y_pred.extend(preds.cpu().numpy())
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total else 0.0
    return acc, y_pred

class CNNBaseline(nn.Module):
    def __init__(self, n, include_pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
        ]
        if include_pool:
            layers.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*layers)
        self.include_pool = include_pool
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
        self.cnn = CNNBaseline(256, include_pool=False)
        self.patch = nn.Conv2d(256, 128, 1, 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, 128))
        self.pos = nn.Parameter(torch.zeros(1, 1024, 128))
        self.tr = nn.Sequential(TransformerEncoder(), TransformerEncoder())
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, n)

    def forward(self, x):
        x = self.cnn.net(x)
        x = self.patch(x).flatten(2).transpose(1, 2)
        B = x.size(0)
        x = torch.cat((self.cls.expand(B, -1, -1), x), 1)
        seq_len = x.size(1)
        pos = self.pos
        if pos.size(1) < seq_len:
            pos = F.pad(pos, (0, 0, 0, seq_len - pos.size(1)))
        else:
            pos = pos[:, :seq_len, :]
        x = x + pos
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
        dataset_ratio=1.0,
        num_workers=4,
        pin_memory=True,
        amp=True,
        channels_last=True
    ):
        self.project_name = project_name
        self.kfolds = kfolds
        self.use_stratified = use_stratified
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_name = model_name
        self.dataset_ratio = dataset_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.amp = amp
        self.channels_last = channels_last
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

import wandb

def ensure_wandb(cfg):
    ratio_label = str(cfg.dataset_ratio).replace(".", "p")
    if wandb.run is None:
        wandb.init(
            project=cfg.project_name,
            name=f"{cfg.model_name}_k{cfg.kfolds}_b{cfg.batch_size}_r{ratio_label}",
            config={
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "kfolds": cfg.kfolds,
                "model": cfg.model_name,
                "dataset_ratio": cfg.dataset_ratio,
                "num_workers": cfg.num_workers,
                "pin_memory": cfg.pin_memory,
                "amp": cfg.amp,
                "channels_last": cfg.channels_last
            },
        )
    else:
        wandb.config.update(
            {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "kfolds": cfg.kfolds,
                "model": cfg.model_name,
                "dataset_ratio": cfg.dataset_ratio,
                "num_workers": cfg.num_workers,
                "pin_memory": cfg.pin_memory,
                "amp": cfg.amp,
                "channels_last": cfg.channels_last
            },
            allow_val_change=True,
        )

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_curve, auc
)
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
def evaluate_test(model, loader, device, amp=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            ctx = (
                torch.amp.autocast(device_type="cuda")
                if amp and device == "cuda"
                else nullcontext()
            )
            with ctx:
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

def log_confusion_matrix(cm, class_names, key_prefix):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    wandb.log({f"{key_prefix}/confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

from sklearn.preprocessing import label_binarize

def log_roc_curve(y_true, y_prob, class_names, key_prefix):
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
    wandb.log({f"{key_prefix}/roc_curve": wandb.Image(fig)})
    plt.close(fig)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        out[:, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture activations.")
        if self.gradients.ndim < 4 or self.activations.ndim < 4:
            raise RuntimeError("GradCAM requires a 4D conv feature map.")
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / cam.max()
        return cam

def get_last_conv_layer(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found for Grad-CAM.")

def log_gradcam(model, loader, device, target_layer, class_names, key_prefix):
    model.eval()
    cam = GradCAM(model, target_layer)
    collected = {}

    for x, y in loader:
        for i in range(x.size(0)):
            class_idx = y[i].item()
            if class_idx in collected:
                continue

            x_i = x[i:i + 1].to(device)
            cam_map = cam.generate(x_i, class_idx).detach().cpu()
            img = x_i.detach().cpu()[0].clamp(0, 1)

            cam_resized = F.interpolate(
                cam_map.unsqueeze(0),
                size=img.shape[1:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-6)

            fig, ax = plt.subplots()
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.imshow(cam_resized.numpy(), cmap="jet", alpha=0.4)
            ax.set_title(class_names[class_idx])
            ax.axis("off")
            wandb.log({f"{key_prefix}/gradcam/{class_names[class_idx]}": wandb.Image(fig)})
            plt.close(fig)

            collected[class_idx] = True
            if len(collected) == len(class_names):
                return

def log_model_artifact(model, cfg, n_classes, fold):
    artifact = wandb.Artifact(
        name=f"{cfg.model_name}-fold{fold}",
        type="model",
        metadata={
            "model_name": cfg.model_name,
            "n_classes": n_classes,
            "kfolds": cfg.kfolds,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "dataset_ratio": cfg.dataset_ratio,
            "fold": fold,
        },
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(
            {
                "model_name": cfg.model_name,
                "n_classes": n_classes,
                "state_dict": model.state_dict(),
            },
            model_path,
        )
        artifact.add_file(model_path, name="model.pth")
        wandb.log_artifact(artifact)

def train_epoch(model, loader, opt, loss_fn, device, amp=False, scaler=None,
                log_prefix=None, step=None):
    model.train()
    losses = []
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        ctx = (
            torch.amp.autocast(device_type="cuda")
            if amp and device == "cuda"
            else nullcontext()
        )
        with ctx:
            logits = model(x)
            loss = loss_fn(logits, y)
        if amp and device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        losses.append(loss.item())
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    acc = correct / total if total else 0.0
    if log_prefix:
        wandb.log(
            {f"{log_prefix}/train_loss": np.mean(losses),
             f"{log_prefix}/acc": acc},
            step=step,
        )
    else:
        wandb.log({"train_loss": np.mean(losses), "acc": acc}, step=step)
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
    ensure_wandb(cfg)
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
        fold_prefix = f"fold_{fold}"

        train_loader = DataLoader(
            Subset(trainval_ds, tr),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            Subset(trainval_ds, va),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )

        model = model_cls(n_classes).to(cfg.device)
        if cfg.channels_last and cfg.device == "cuda":
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        loss_fn = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(
            "cuda", enabled=cfg.amp and cfg.device == "cuda"
        )

        for ep in range(cfg.epochs):
            loss = train_epoch(
                model, train_loader, opt, loss_fn, cfg.device,
                amp=cfg.amp, scaler=scaler, log_prefix=fold_prefix, step=ep
            )
            acc, _ = evaluate(model, val_loader, cfg.device, amp=cfg.amp)
            wandb.log({f"{fold_prefix}/val_accuracy": acc}, step=ep)

        metrics, cm, y_true, y_prob = evaluate_test(
            model, test_loader, cfg.device, amp=cfg.amp
        )

        wandb.log({f"{fold_prefix}/{k}": v for k, v in metrics.items()})
        log_confusion_matrix(cm, test_ds.classes, fold_prefix)
        log_roc_curve(y_true, y_prob, test_ds.classes, fold_prefix)

        # Grad-CAM (last CNN layer)
        log_gradcam(
            model,
            test_loader,
            cfg.device,
            target_layer=get_last_conv_layer(model),
            class_names=test_ds.classes,
            key_prefix=fold_prefix,
        )

        log_model_artifact(model, cfg, n_classes, fold)

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
        [base_cfg.epochs] if base_cfg else [10]
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
                                    dataset_ratio=ratio,
                                    num_workers=base_cfg.num_workers if base_cfg else 4,
                                    pin_memory=base_cfg.pin_memory if base_cfg else True,
                                    amp=base_cfg.amp if base_cfg else True,
                                    channels_last=base_cfg.channels_last if base_cfg else True
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
    parser.add_argument("--batch_size", dest="batch_size", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--dataset-ratio", type=float, default=1.0)
    parser.add_argument("--dataset_ratio", dest="dataset_ratio", type=float)
    parser.add_argument("--model_name", dest="model_name",
                        choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--no-stratified", action="store_true",
                        help="Disable stratified splitting.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
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
        dataset_ratio=args.dataset_ratio,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        amp=not args.no_amp,
        channels_last=not args.no_channels_last
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
