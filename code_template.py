import argparse
import json
import logging
import os
import random
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
def build_model(backbone, num_classes, pretrained):
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


def count_class_occurrences(dataframe: pd.DataFrame) -> Dict[str, int]:
    return {disease: int(dataframe[disease].sum()) for disease in DISEASE_NAMES}


def balance_training_data(
    train_csv: str,
    train_image_dir: str,
    classes_to_balance: tuple = ("G", "A"),
    random_seed: int = 42,
) -> Dict[str, Any]:
    original_dataframe = pd.read_csv(train_csv)
    counts = count_class_occurrences(original_dataframe)
    logger.info("Training set counts before balancing: %s", counts)

    target_count = max(counts.values()) if counts else 0
    if target_count == 0:
        logger.warning("Training set appears to be empty; skipping balancing.")
        return {"original_df": original_dataframe, "new_files": []}

    rng = random.Random(random_seed)
    existing_ids = set(original_dataframe["id"].tolist())
    new_rows = []
    new_files = []

    for disease in classes_to_balance:
        current_count = counts.get(disease, 0)
        deficit = target_count - current_count
        if deficit <= 0:
            continue
        candidates = original_dataframe[original_dataframe[disease] == 1]
        if candidates.empty:
            logger.warning("No candidate images found for class %s; skipping.", disease)
            continue
        for index in range(deficit):
            row = candidates.iloc[rng.randrange(len(candidates))]
            image_name = row["id"]
            image_path = os.path.join(train_image_dir, image_name)
            if not os.path.exists(image_path):
                logger.warning("Missing image %s; skipping augmentation.", image_path)
                continue
            image = Image.open(image_path).convert("RGB")
            flip_type = rng.choice(["h", "v"])
            if flip_type == "h":
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

            base, ext = os.path.splitext(image_name)
            counter = index
            new_name = f"{base}_flip_{flip_type}_{counter}{ext}"
            while new_name in existing_ids or os.path.exists(os.path.join(train_image_dir, new_name)):
                counter += 1
                new_name = f"{base}_flip_{flip_type}_{counter}{ext}"

            new_path = os.path.join(train_image_dir, new_name)
            image.save(new_path)
            existing_ids.add(new_name)
            new_row = row.copy()
            new_row["id"] = new_name
            new_rows.append(new_row)
            new_files.append(new_path)

    if new_rows:
        augmented = pd.concat([original_dataframe, pd.DataFrame(new_rows)], ignore_index=True)
        augmented = augmented[["id"] + DISEASE_NAMES]
        augmented.to_csv(train_csv, index=False)
        counts_after = count_class_occurrences(augmented)
        logger.info("Training set counts after balancing: %s", counts_after)
    else:
        logger.info("No balancing required; training set already balanced.")

    return {"original_df": original_dataframe, "new_files": new_files}


def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


# ========================
# model training and val
# ========================
def train_one_backbone(
    backbone,
    tuning_method,
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
    logger.info("Training device: %s", describe_device(device))

    balancing_result = balance_training_data(train_csv, train_image_dir)

    # transforms
    transform = build_transforms(img_size)

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model
    model = build_model(backbone, num_classes=num_classes, pretrained=False).to(device)

    for p in model.parameters():
        p.requires_grad = False
    if tuning_method == "classifier_only":
        if backbone == "resnet18":
            for p in model.fc.parameters():
                p.requires_grad = True
        elif backbone == "efficientnet":
            for p in model.classifier.parameters():
                p.requires_grad = True
    elif tuning_method == "full":
        for p in model.parameters():
            p.requires_grad = True
    log_model_details(model, backbone)

    # loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        load_checkpoint(model, pretrained_backbone, device)

    try:
        initial_patience = 0
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
                initial_patience = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best model for {backbone} at {ckpt_path}")
            else:
                initial_patience += 1
                if initial_patience >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1} for {backbone}")
                    break
    finally:
        original_df = balancing_result.get("original_df")
        new_files = balancing_result.get("new_files", [])
        if original_df is not None:
            original_df.to_csv(train_csv, index=False)
        for file_path in new_files:
            if os.path.exists(file_path):
                os.remove(file_path)

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
    img_size,
    batch_size,
    num_classes,
):
    device = get_device()
    logger.info("Prediction device: %s", describe_device(device))
    transform = build_transforms(img_size)

    predict_ds = RetinaPredictDataset(image_csv, image_dir, transform)
    predict_loader = DataLoader(predict_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(backbone, num_classes=num_classes, pretrained=False).to(device)
    log_model_details(model, backbone)
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
    tuning_method: str = "full"
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

    checkpoint_path = config.checkpoint_path
    if config.do_train:
        checkpoint_path = train_one_backbone(
            backbone=config.backbone,
            tuning_method=config.tuning_method,
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
    elif checkpoint_path is None and config.pretrained_backbone is None:
        raise ValueError("Provide checkpoint_path or pretrained_backbone when training is disabled.")

    if not config.do_train and checkpoint_path is None:
        checkpoint_path = config.pretrained_backbone

    if config.do_test:
        transform = build_transforms(config.img_size)
        test_ds = RetinaMultiLabelDataset(config.test_csv, config.test_image_dir, transform)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
        model = build_model(config.backbone, num_classes=config.num_classes, pretrained=False).to(device)
        log_model_details(model, config.backbone)
        load_checkpoint(model, checkpoint_path, device)
        evaluate_model(model, test_loader, device, config.backbone)

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
