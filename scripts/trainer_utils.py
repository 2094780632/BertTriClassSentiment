"""
共享训练工具：数据集加载、训练循环、评估函数
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from tqdm import tqdm
from config import MODEL_NAME, MAX_LEN, DEVICE


# ==================== 数据集 ====================

class SentimentDataset(Dataset):
    """BERT 文本分类 Dataset"""

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


# ==================== 评估指标 ====================

def compute_metrics(labels, preds, probs=None):
    """计算二分类各项指标"""
    labels = np.array(labels)
    preds = np.array(preds)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


# ==================== 训练函数 ====================

def train_stage(
    model,
    train_loader,
    val_loader,
    stage_name: str,
    epochs: int = 5,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    save_dir: str = None,
):
    """
    通用的二分类 BERT 训练流程。
    
    Args:
        model: BertBinaryClassifier 实例
        train_loader / val_loader: DataLoader
        stage_name: "stage1" 或 "stage2"，用于日志
        epochs / lr / ... : 训练超参数
        save_dir: 模型保存目录
    """
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器 & 调度器
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"[{stage_name}] Epoch {epoch+1}/{epochs} Train")
        for step, batch in enumerate(pbar):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["label"].to(device),
            }
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # --- Val ---
        val_metrics = evaluate_model(model, val_loader, device)
        history["val_f1"].append(val_metrics["f1"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"[{stage_name}] Epoch {epoch+1} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        # --- Save Best ---
        if val_metrics["f1"] > best_val_f1 and save_dir:
            best_val_f1 = val_metrics["f1"]
            model.save_pretrained(save_dir)
            tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
            tokenizer.save_pretrained(save_dir)
            print(f"[{stage_name}] ✅ 最佳模型已保存 (F1={best_val_f1:.4f}) → {save_dir}")

    return model, history, best_val_f1


def evaluate_model(model, data_loader, device):
    """评估二分类模型，返回指标字典"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs["logits"].view(-1))
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].long().tolist())
    return compute_metrics(all_labels, all_preds)


def load_data_for_stage(csv_path: str):
    """从 CSV 加载文本和标签"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    # 兼容 stage1 / stage2 的列名
    if "stage1_label" in df.columns:
        label_col = "stage1_label"
    elif "stage2_label" in df.columns:
        label_col = "stage2_label"
    else:
        label_col = "label"
    return df["review"].tolist(), df[label_col].tolist()
