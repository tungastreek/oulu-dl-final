import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


DISEASE_NAMES = ["D", "G", "A"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ========================
# Dataset preparation
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
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
def build_model(backbone="resnet18", num_classes=3, pretrained=True):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model


def build_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


# ========================
# model training and val
# ========================
def train_one_backbone(
    backbone,
    train_csv,
    val_csv,
    train_image_dir,
    val_image_dir,
    epochs=10,
    batch_size=32,
    lr=1e-4,
    img_size=256,
    save_dir="checkpoints",
    pretrained_backbone=None,
):
    device = get_device()

    # transforms
    transform = build_transforms(img_size)

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    for p in model.parameters():
        p.requires_grad = True

    # loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        load_checkpoint(model, pretrained_backbone, device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{backbone}] Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model for {backbone} at {ckpt_path}")

    return ckpt_path


def evaluate_model(model, test_loader, device, backbone):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = torch.tensor(y_true).numpy()
    y_pred = torch.tensor(y_pred).numpy()

    for i, disease in enumerate(DISEASE_NAMES):  # compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall = recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"{disease} Results [{backbone}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}")


# ========================
# prediction
# ========================
def predict_from_images(
    backbone,
    checkpoint_path,
    image_csv,
    image_dir,
    output_csv,
    img_size=256,
    batch_size=32,
    num_classes=3,
):
    device = get_device()
    transform = build_transforms(img_size)

    predict_ds = RetinaPredictDataset(image_csv, image_dir, transform)
    predict_loader = DataLoader(predict_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(backbone, num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, checkpoint_path, device)
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
    print(f"Saved predictions to {output_csv}")


@dataclass
class RunnerConfig:
    backbone: str = "resnet18"
    num_classes: int = 3
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "offsite_test.csv"
    train_image_dir: str = "./images/train"
    val_image_dir: str = "./images/val"
    test_image_dir: str = "./images/offsite_test"
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-5
    img_size: int = 256
    save_dir: str = "checkpoints"
    pretrained_backbone: Optional[str] = "./pretrained_backbone/ckpt_resnet18_ep50.pt"
    checkpoint_path: Optional[str] = None
    do_train: bool = True
    do_test: bool = True
    do_predict: bool = False
    predict_input_csv: Optional[str] = None
    predict_image_dir: Optional[str] = None
    predict_output_csv: str = "predictions.csv"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunnerConfig":
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in data.items() if key in known_fields}
        return cls(**filtered)


def run_pipeline(config: RunnerConfig):
    checkpoint_path = config.checkpoint_path
    if config.do_train:
        checkpoint_path = train_one_backbone(
            config.backbone,
            config.train_csv,
            config.val_csv,
            config.train_image_dir,
            config.val_image_dir,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            img_size=config.img_size,
            save_dir=config.save_dir,
            pretrained_backbone=config.pretrained_backbone,
        )
    elif checkpoint_path is None and config.pretrained_backbone is None:
        raise ValueError("Provide checkpoint_path or pretrained_backbone when training is disabled.")

    if not config.do_train and checkpoint_path is None:
        checkpoint_path = config.pretrained_backbone

    if config.do_test:
        device = get_device()
        transform = build_transforms(config.img_size)
        test_ds = RetinaMultiLabelDataset(config.test_csv, config.test_image_dir, transform)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
        model = build_model(config.backbone, num_classes=config.num_classes, pretrained=False).to(device)
        load_checkpoint(model, checkpoint_path, device)
        evaluate_model(model, test_loader, device, config.backbone)

    if config.do_predict:
        if config.predict_input_csv is None or config.predict_image_dir is None:
            raise ValueError("predict_input_csv and predict_image_dir are required for prediction.")
        predict_from_images(
            config.backbone,
            checkpoint_path,
            config.predict_input_csv,
            config.predict_image_dir,
            config.predict_output_csv,
            img_size=config.img_size,
            batch_size=config.batch_size,
            num_classes=config.num_classes,
        )


# ========================
# main
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test/predict retina models.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    parser.add_argument("--do-train", action="store_true", help="Enable training.")
    parser.add_argument("--no-train", action="store_true", help="Disable training.")
    parser.add_argument("--do-test", action="store_true", help="Enable testing.")
    parser.add_argument("--no-test", action="store_true", help="Disable testing.")
    parser.add_argument("--do-predict", action="store_true", help="Enable prediction.")
    parser.add_argument("--no-predict", action="store_true", help="Disable prediction.")
    parser.add_argument("--checkpoint-path", type=str, help="Path to a checkpoint to load.")
    parser.add_argument("--pretrained-backbone", type=str, help="Path to pretrained backbone weights.")
    parser.add_argument("--predict-input-csv", type=str, help="CSV listing images to predict.")
    parser.add_argument("--predict-image-dir", type=str, help="Directory containing prediction images.")
    parser.add_argument("--predict-output-csv", type=str, help="Output CSV path for predictions.")
    args = parser.parse_args()

    config_data: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)

    config = RunnerConfig.from_dict(config_data)

    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path
    if args.pretrained_backbone:
        config.pretrained_backbone = args.pretrained_backbone
    if args.predict_input_csv:
        config.predict_input_csv = args.predict_input_csv
    if args.predict_image_dir:
        config.predict_image_dir = args.predict_image_dir
    if args.predict_output_csv:
        config.predict_output_csv = args.predict_output_csv

    if args.do_train:
        config.do_train = True
    if args.no_train:
        config.do_train = False
    if args.do_test:
        config.do_test = True
    if args.no_test:
        config.do_test = False
    if args.do_predict:
        config.do_predict = True
    if args.no_predict:
        config.do_predict = False

    run_pipeline(config)
