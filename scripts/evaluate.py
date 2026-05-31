"""
评估脚本：
1. 加载 Stage1 + Stage2 模型，级联推理
2. 三分类整体评估（Accuracy, F1, Precision, Recall, Confusion Matrix）
3. 各阶段独立评估
4. 可选单条测试
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from tabulate import tabulate

from config import (
    MODEL_NAME, MAX_LEN, EVAL_BATCH_SIZE,
    STAGE1_MODEL_DIR, STAGE2_MODEL_DIR,
    PROCESSED_DIR, LABEL_NAMES,
)
from model import BertBinaryClassifier, CascadeSentimentModel
from trainer_utils import SentimentDataset


def load_cascade_model(device):
    """加载级联模型（Stage1 + Stage2）"""
    print("加载 Stage1 模型...")
    stage1 = BertBinaryClassifier.from_pretrained(STAGE1_MODEL_DIR)
    print("加载 Stage2 模型...")
    stage2 = BertBinaryClassifier.from_pretrained(STAGE2_MODEL_DIR)

    cascade = CascadeSentimentModel(stage1, stage2)
    cascade.to(device)
    cascade.eval()
    print("✅ 级联模型加载完成\n")
    return cascade, stage1, stage2


def evaluate_cascade(cascade, test_loader, device):
    """
    级联推理 + 三分类评估
    """
    all_preds = []
    all_labels = []
    stage1_correct = 0
    stage2_correct = 0
    stage2_total = 0

    for batch in test_loader:
        labels = batch["label"].long().tolist()  # 三分类标签: 0/1/2
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }

        # 级联预测
        preds = cascade.predict(inputs, threshold=0.5)

        all_preds.extend(preds)
        all_labels.extend(labels)

        # 统计各阶段正确率
        for p, l in zip(preds, labels):
            # Stage1: 判断中性是否正确
            is_neutral_true = (l == 1)
            is_neutral_pred = (p == 1)
            if is_neutral_true == is_neutral_pred:
                stage1_correct += 1
            # Stage2: 对非中性样本判断正负面是否正确
            if not is_neutral_true:  # 真实非中性
                stage2_total += 1
                if p == l:  # 级联最终结果正确
                    stage2_correct += 1

    # 三分类指标
    target_names = [LABEL_NAMES[i] for i in range(3)]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    return metrics, report, cm, stage1_correct / len(all_labels), (
        stage2_correct / stage2_total if stage2_total > 0 else 0
    )


def evaluate_stage_individually(stage_model, test_loader, device, stage_name, label_key):
    """独立评估某一阶段的二分类性能"""
    stage_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].long().tolist()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = stage_model(**inputs)
            probs = torch.sigmoid(outputs["logits"].view(-1))
            preds = (probs >= 0.5).long().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n--- {stage_name} 独立评估 ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def predict_single(cascade, tokenizer, text, device):
    """单条文本预测"""
    encoding = tokenizer(
        text, max_length=MAX_LEN, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    inputs = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }
    result = cascade.predict_with_probs(inputs, threshold=0.5)
    pred = result["predictions"][0]
    probs = result["probs"][0]
    return pred, probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 加载模型
    cascade, stage1, stage2 = load_cascade_model(device)
    tokenizer = BertTokenizer.from_pretrained(STAGE1_MODEL_DIR)

    # ========== 1. 级联三分类评估 ==========
    print("=" * 60)
    print("级联模型三分类评估")
    print("=" * 60)

    test_df = pd.read_csv(f"{PROCESSED_DIR}/test.csv")
    test_texts = test_df["review"].tolist()
    test_labels = test_df["label"].tolist()

    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    metrics, report, cm, stage1_acc, stage2_acc = evaluate_cascade(cascade, test_loader, device)

    print(f"\n📊 整体三分类指标:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")

    print(f"\n📊 各阶段准确率:")
    print(f"  Stage1 (中性识别):  {stage1_acc:.4f}")
    print(f"  Stage2 (正负面识别): {stage2_acc:.4f}")

    print(f"\n📋 分类报告:")
    print(report)

    print("📋 混淆矩阵:")
    cm_df = pd.DataFrame(
        cm,
        index=[f"真实-{LABEL_NAMES[i]}" for i in range(3)],
        columns=[f"预测-{LABEL_NAMES[i]}" for i in range(3)],
    )
    print(tabulate(cm_df, headers="keys", tablefmt="grid"))

    # ========== 2. 各阶段独立评估 ==========
    print("\n" + "=" * 60)
    print("各阶段独立评估")
    print("=" * 60)

    # Stage1 独立评估：中性 vs 非中性
    stage1_test_df = pd.read_csv(f"{PROCESSED_DIR}/stage1_test.csv")
    s1_texts = stage1_test_df["review"].tolist()
    s1_labels = stage1_test_df["stage1_label"].tolist()
    s1_dataset = SentimentDataset(s1_texts, s1_labels, tokenizer, MAX_LEN)
    s1_loader = DataLoader(s1_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    evaluate_stage_individually(stage1, s1_loader, device, "Stage1 (中性 vs 非中性)", "stage1_label")

    # Stage2 独立评估：负面 vs 正面
    stage2_test_df = pd.read_csv(f"{PROCESSED_DIR}/stage2_test.csv")
    s2_texts = stage2_test_df["review"].tolist()
    s2_labels = stage2_test_df["stage2_label"].tolist()
    s2_dataset = SentimentDataset(s2_texts, s2_labels, tokenizer, MAX_LEN)
    s2_loader = DataLoader(s2_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    evaluate_stage_individually(stage2, s2_loader, device, "Stage2 (负面 vs 正面)", "stage2_label")

    # ========== 3. 单条测试 ==========
    print("\n" + "=" * 60)
    print("单条测试样例")
    print("=" * 60)

    samples = [
        "酒店环境不错，服务也很热情，下次还会再来。",
        "太差了，房间又脏又小，前台态度恶劣，绝不推荐。",
        "一般般吧，不好不坏，没什么特别的感觉。",
        "早餐种类还行，但是房间隔音不太好，勉强接受。",
        "位置方便，价格实惠，整体感觉很满意！",
    ]
    for text in samples:
        pred, probs = predict_single(cascade, tokenizer, text, device)
        print(f"文本: {text}")
        print(f"  预测: {LABEL_NAMES[pred]} | 概率 [负:{probs[0]:.3f} 中:{probs[1]:.3f} 正:{probs[2]:.3f}]")
        print()

    print("=" * 60)
    print("✅ 评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
