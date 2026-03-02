#%% Import libraries
import os
import sys
import csv
import json
import argparse
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_category_files, split_val_test, train_dir, val_test_dir

#%% Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

#%% Reproducibility
seed = 36
def set_seed(seed=seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")

#%% Constants
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(project_root, "phobert-v2")
best_model = os.path.join(output_dir, "best_model")
history = os.path.join(output_dir, "training_history.csv")
log_path = os.path.join(output_dir, "training_log.txt")
phobert = "vinai/phobert-base-v2"

#%% Dataset
class VNNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


#%% Callbacks
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

class CheckpointCallback:
    def __init__(self, save_dir, tokenizer):
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.best_loss = float("inf")

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            os.makedirs(self.save_dir, exist_ok=True)
            model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)
            logger.info(f"  Best model checkpoint saved → {self.save_dir}")
            return True
        return False

class HistoryCallback:
    HEADERS = ["epoch", "split", "loss", "acc", "precision", "recall", "f1", "time_s"]

    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.HEADERS).writeheader()

    def step(self, epoch, split, loss, acc, precision, recall, f1, time_s=""):
        row = {
            "epoch": epoch,
            "split": split,
            "loss": round(loss, 6),
            "acc": round(acc, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "time_s": time_s,
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writerow(row)


#%% Training helpers
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, loss_fn):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for step, batch in enumerate(loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step % 50 == 0 or step == len(loader):
            logger.info(
                f"  Epoch {epoch} | step {step}/{len(loader)} "
                f"| loss={total_loss / step:.4f} | acc={correct / total:.4f}"
            )

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        total_loss += loss_fn(outputs.logits, labels).item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), correct / total, all_preds, all_labels


def compute_metrics(preds, labels):
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return precision, recall, f1


def save_confusion_matrix(preds, labels, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"  Confusion matrix saved → {save_path}")


#%% Main
def parse_args():
    parser = argparse.ArgumentParser(description="PhoBERT Vietnamese News Classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(output_dir, exist_ok=True)

    set_seed(seed)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Training log → {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading data …")
    train_df, label_map = load_category_files(train_dir)
    val_test_df, _ = load_category_files(val_test_dir)
    val_df, test_df = split_val_test(val_test_df)

    num_labels = len(label_map)
    logger.info(
        f"Train: {len(train_df):,}  Val: {len(val_df):,}  "
        f"Test: {len(test_df):,}  Classes: {num_labels}"
    )
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=train_df["label"].values,
    )

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    logger.info(f"Class weights: { {k: round(float(v), 4) for k, v in zip(label_map.keys(), class_weights)} }")

    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Label map saved → {label_map_path}")

    logger.info(f"Loading tokenizer: {phobert}")
    tokenizer = AutoTokenizer.from_pretrained(phobert)

    logger.info("Tokenising datasets (this may take a few minutes) …")
    train_dataset = VNNewsDataset(train_df["text"], train_df["label"], tokenizer, args.max_len)
    val_dataset = VNNewsDataset(val_df["text"], val_df["label"], tokenizer, args.max_len)
    test_dataset = VNNewsDataset(test_df["text"], test_df["label"], tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Loading model: {phobert}")
    model = AutoModelForSequenceClassification.from_pretrained(
        phobert,
        num_labels=num_labels,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"Total training steps: {total_steps:,}  Warmup steps: {warmup_steps:,}")

    early_stopping = EarlyStopping(patience=args.patience)
    checkpoint_cb = CheckpointCallback(save_dir=best_model, tokenizer=tokenizer)
    history_cb = HistoryCallback(csv_path=history)

    logger.info("=" * 60)
    logger.info("Starting training …")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        logger.info(f"\n── Epoch {epoch}/{args.epochs} ──────────────────────────────")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, loss_fn)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device, loss_fn)
        val_precision, val_recall, val_f1 = compute_metrics(val_preds, val_labels)
        epoch_time = time.time() - t0

        logger.info(
            f"  [Summary] train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_precision={val_precision:.4f}  val_recall={val_recall:.4f}  val_f1={val_f1:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        checkpoint_cb.step(val_loss, model)

        history_cb.step(epoch, "train", train_loss, train_acc, 0.0, 0.0, 0.0, round(epoch_time, 1))
        history_cb.step(epoch, "val", val_loss, val_acc, val_precision, val_recall, val_f1)

        if early_stopping.step(val_loss):
            logger.info(
                f"\n🛑 Early stopping triggered after epoch {epoch} "
                f"(no val_loss improvement for {args.patience} consecutive epochs)."
            )
            break

    logger.info(f"Training history → {history}")

    logger.info("\nEvaluating on test set …")
    best_model_loaded = AutoModelForSequenceClassification.from_pretrained(best_model)
    best_model_loaded.to(device)
    test_loss, test_acc, test_preds, test_labels = evaluate(best_model_loaded, test_loader, device, loss_fn)
    test_precision, test_recall, test_f1 = compute_metrics(test_preds, test_labels)
    logger.info(
        f"  Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}  "
        f"Test precision: {test_precision:.4f}  Test recall: {test_recall:.4f}  Test F1: {test_f1:.4f}"
    )

    class_names = list(label_map.keys())
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix(test_preds, test_labels, class_names, cm_path)

    history_cb.step("test", "test", test_loss, test_acc, test_precision, test_recall, test_f1)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Best model: {best_model}")
    logger.info(f"  History:    {history}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
