import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


DISEASE_NAMES = ["D", "G", "A"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_details(model: nn.Module, backbone: str) -> None:
    trainable_params = count_trainable_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model backbone: %s | Trainable params: %d | Total params: %d",
        backbone,
        trainable_params,
        total_params,
    )

# =======================
# Config class
# =======================
@dataclass
class RunnerConfig:
    backbone: str = "resnet18"
    num_classes: int = 3
    tuning_method: str = "full"
    loss_function: str = "bce"
    attention: str = "none"
    label_smoothing_epsilon: float = 0.0
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "offsite_test.csv"
    train_image_dir: str = "./images/train"
    val_image_dir: str = "./images/val"
    test_image_dir: str = "./images/offsite_test"
    num_workers: int = 0
    early_stopping_patience: int = 15
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-4
    img_size: int = 256
    save_dir: str = "checkpoints"
    pretrained_backbone: Optional[str] = "./pretrained_backbone/ckpt_resnet18_ep50.pt"
    do_train: bool = True
    do_test: bool = True
    do_predict: bool = False
    predict_input_csv: Optional[str] = None
    predict_image_dir: Optional[str] = None
    predict_output_csv: str = "predictions.csv"
    offline_result_csv: str = "offline_result.csv"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunnerConfig":
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in data.items() if key in known_fields}
        return cls(**filtered)


# =======================
# Custom Loss Functions
# =======================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=inputs.dtype, device=inputs.device)

        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        return loss.mean()


# ============================
# Custom Attention Mechanisms
# ============================
class SE(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()
        hidden = max(1, channel // reduction_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y
    

class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.size()
        x_reshaped = x.view(b, c, h * w).permute(0, 2, 1)
        attn_output, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
        return attn_output


class AttnThenPool(nn.Module):
    def __init__(self, attn: nn.Module, pool: nn.Module):
        super().__init__()
        self.attn = attn
        self.pool = pool

    def forward(self, x):
        x = self.attn(x)
        x = self.pool(x)
        return x


# ========================
# Dataset preparation
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, include_id: bool = False):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.include_id = include_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row.iloc[0]
        img_path = os.path.join(self.image_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        if self.include_id:
            return img, labels, image_name
        return img, labels


class RetinaPredictDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row.iloc[0]
        img_path = os.path.join(self.image_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return image_name, img


# ========================
# build model
# ========================
def build_model(backbone, num_classes, pretrained, attention):
    # resnet18
    if backbone == "resnet18":
        # weights
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        model = models.resnet18(weights=weights)
        # head
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # attention
        if attention != "none":
            if attention == "se":
                attn = SE(channel=512, reduction_ratio=16)
            elif attention == "mha":
                attn = MHA(embed_dim=512, num_heads=8)
            else:
                raise ValueError("Unsupported attention mechanism")
            model.avgpool = AttnThenPool(attn, model.avgpool)

    # efficientnet
    elif backbone == "efficientnet":
        # weights
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
        model = models.efficientnet_b0(weights=weights)
        # head
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        # attention
        if attention != "none":
            raise ValueError("Attention mechanisms not supported for EfficientNet backbone")

    # Swin Transformer
    elif backbone == "swin":
        # weights
        weights = models.Swin_V2_T_Weights.DEFAULT
        model = models.swin_v2_t(weights=weights)
        # head
        model.head = nn.Linear(model.head.in_features, num_classes)
        # attention
        if attention != "none":
            raise ValueError("Attention mechanisms not supported for Swin Transformer backbone")

    else:
        raise ValueError("Unsupported backbone")
    return model


def build_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_checkpoint(model, checkpoint_path, device, strict):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=strict)


# ========================
# model training and val
# ========================
def train_one_backbone(
    backbone,
    tuning_method,
    loss_function,
    attention,
    label_smoothing_epsilon,
    num_classes,
    train_csv,
    val_csv,
    train_image_dir,
    val_image_dir,
    num_workers,
    epochs,
    early_stopping_patience,
    batch_size,
    lr,
    weight_decay,
    img_size,
    save_dir,
    pretrained_backbone,
):
    device = get_device()
    # run_pipeline() already logs device at INFO; keep this non-redundant
    logger.debug("Training device: %s", describe_device(device))

    # transforms
    transform = build_transforms(img_size)

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # calculate class weights based on train_csv
    train_df = pd.read_csv(train_csv)
    pos = train_df.iloc[:, 1:].sum(axis=0).to_numpy(dtype=np.float32)
    n = float(len(train_df))
    neg = (n - pos).astype(np.float32)

    eps = 1e-6
    pos_weight = torch.tensor(neg / (pos + eps), dtype=torch.float32, device=device).clamp(min=0.0)  # wbce
    focal_alpha = torch.tensor(neg / (n + eps), dtype=torch.float32, device=device).clamp(0.0, 1.0)  # focal

    logger.info("pos_weight (wbce) D,G,A: %s", pos_weight.detach().cpu().numpy().round(4).tolist())
    logger.info("alpha (focal)     D,G,A: %s", focal_alpha.detach().cpu().numpy().round(4).tolist())

    # model
    model = build_model(backbone, num_classes=num_classes, pretrained=False, attention=attention).to(device)

    for p in model.parameters():
        p.requires_grad = False
    if tuning_method == "classifier_only":
        if backbone == "resnet18":
            for p in model.fc.parameters():
                p.requires_grad = True
        elif backbone == "efficientnet":
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif backbone == "swin":
            for p in model.head.parameters():
                p.requires_grad = True
    elif tuning_method == "full":
        for p in model.parameters():
            p.requires_grad = True
    log_model_details(model, backbone)

    # loss & optimizer
    if loss_function == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=2)
    elif loss_function == "wbce":
        criterion = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
    else:
        raise ValueError("Unsupported loss function")
    if weight_decay != 0:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # training
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    best_f1 = float("-inf")

    # load pretrained backbone
    if pretrained_backbone is not None:
        load_checkpoint(model, pretrained_backbone, device, strict=(attention == "none"))

    no_improve_patience = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # label smoothing
            if label_smoothing_epsilon > 0.0:
                labels_smoothed = labels * (1.0 - label_smoothing_epsilon) + (1.0 - labels) * label_smoothing_epsilon 
                labels = labels_smoothed
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).to(torch.int32)

                y_true_val.append(labels.detach().cpu().numpy())
                y_pred_val.append(preds.detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)

        y_true_val = np.concatenate(y_true_val, axis=0)
        y_pred_val = np.concatenate(y_pred_val, axis=0)
        val_f1_macro = f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)

        logger.info(
            "[%s] Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val macro-F1: %.4f",
            backbone,
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            val_f1_macro,
        )

        # save best-by-f1 (single checkpoint)
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved best-by-f1 model for %s at %s (val_f1=%.4f)", backbone, ckpt_path, val_f1_macro)
            no_improve_patience = 0
        else:
            no_improve_patience += 1
            if no_improve_patience >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d for %s (no improvement in val macro-F1 for %d epochs)",
                    epoch + 1,
                    backbone,
                    early_stopping_patience,
                )
                break

    return ckpt_path


def evaluate_model(model, test_loader, device, backbone):
    model.eval()
    y_true, y_pred = [], []
    offline_results = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                imgs, labels, image_names = batch
            else:
                imgs, labels = batch
                image_names = None

            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

            if image_names is not None:
                for image_name, prob, pred in zip(image_names, probs, preds):
                    row = {"id": image_name}
                    for idx, disease in enumerate(DISEASE_NAMES):
                        row[f"{disease}_prob"] = float(prob[idx])
                        row[disease] = int(pred[idx])
                    offline_results.append(row)

    y_true = torch.tensor(np.array(y_true)).numpy()
    y_pred = torch.tensor(np.array(y_pred)).numpy()

    for i, disease in enumerate(DISEASE_NAMES):  # compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall = recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        logger.info(
            "%s Results [%s] | Acc: %.4f | Prec: %.4f | Recall: %.4f | F1: %.4f | Kappa: %.4f",
            disease,
            backbone,
            acc,
            precision,
            recall,
            f1,
            kappa,
        )

    return offline_results

def predict_from_images(
    backbone,
    checkpoint_path,
    image_csv,
    image_dir,
    output_csv,
    img_size,
    batch_size,
    num_classes,
    attention,
):
    device = get_device()
    logger.debug("Prediction device: %s", describe_device(device))
    transform = build_transforms(img_size)

    predict_ds = RetinaPredictDataset(image_csv, image_dir, transform)
    predict_loader = DataLoader(predict_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(backbone, num_classes=num_classes, pretrained=False, attention=attention).to(device)
    log_model_details(model, backbone)
    load_checkpoint(model, checkpoint_path, device, strict=(attention == "none"))
    model.eval()

    results = []
    with torch.no_grad():
        for image_names, imgs in predict_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            for image_name, prob, pred in zip(image_names, probs, preds):
                row = {"id": image_name}
                for idx, disease in enumerate(DISEASE_NAMES):
                    row[disease] = int(pred[idx])
                results.append(row)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    logger.info("Saved predictions to %s", output_csv)


def run_pipeline(config: RunnerConfig):
    enabled_stages = []
    if config.do_train:
        enabled_stages.append("train")
    if config.do_test:
        enabled_stages.append("test")
    if config.do_predict:
        enabled_stages.append("predict")

    if config.do_train and config.do_predict:
        mode = "train & predict"
    elif (not config.do_train) and config.do_predict:
        mode = "predict only"
    elif enabled_stages:
        mode = " & ".join(enabled_stages)
    else:
        mode = "none"

    device = get_device()
    logger.info("Run mode: %s", mode)
    logger.info("Selected device: %s", describe_device(device))
    logger.info("Backbone: %s", config.backbone)

    checkpoint_path: Optional[str] = None
    if config.do_train:
        checkpoint_path = train_one_backbone(
            backbone=config.backbone,
            tuning_method=config.tuning_method,
            loss_function=config.loss_function,
            attention=config.attention,
            label_smoothing_epsilon=config.label_smoothing_epsilon,
            num_classes=config.num_classes,
            train_csv=config.train_csv,
            val_csv=config.val_csv,
            train_image_dir=config.train_image_dir,
            val_image_dir=config.val_image_dir,
            num_workers=config.num_workers,
            epochs=config.epochs,
            early_stopping_patience=config.early_stopping_patience,
            batch_size=config.batch_size,
            lr=config.lr,
            weight_decay=config.weight_decay,
            img_size=config.img_size,
            save_dir=config.save_dir,
            pretrained_backbone=config.pretrained_backbone,
        )
    else:
        if config.pretrained_backbone is None:
            raise ValueError("Provide pretrained_backbone when training is disabled.")
        checkpoint_path = config.pretrained_backbone

    if config.do_test:
        transform = build_transforms(config.img_size)
        test_ds = RetinaMultiLabelDataset(config.test_csv, config.test_image_dir, transform, include_id=True)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
        model = build_model(config.backbone, num_classes=config.num_classes, pretrained=False, attention=config.attention).to(device)
        log_model_details(model, config.backbone)
        load_checkpoint(model, checkpoint_path, device, strict=(config.attention=="none"))
        offline_results = evaluate_model(model, test_loader, device, config.backbone)
        if offline_results:
            columns = ["id"]
            for disease in DISEASE_NAMES:
                columns.extend([f"{disease}_prob", disease])
            df = pd.DataFrame(offline_results)
            df = df.reindex(columns=columns)
            df.to_csv(config.offline_result_csv, index=False)
            logger.info("Saved offline predictions with probabilities to %s", config.offline_result_csv)

    if config.do_predict:
        if config.predict_input_csv is None or config.predict_image_dir is None:
            raise ValueError("predict_input_csv and predict_image_dir are required for prediction.")
        predict_from_images(
            backbone=config.backbone,
            checkpoint_path=checkpoint_path,
            image_csv=config.predict_input_csv,
            image_dir=config.predict_image_dir,
            output_csv=config.predict_output_csv,
            img_size=config.img_size,
            batch_size=config.batch_size,
            num_classes=config.num_classes,
            attention=config.attention,
        )


# ========================
# main
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test/predict retina models.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    args = parser.parse_args()

    config_data: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)

    config = RunnerConfig.from_dict(config_data)

    run_pipeline(config)
