"""
Stage2 训练脚本：负面 vs 正面 二分类

仅在非中性样本上训练，判断是负面还是正面。
- 负面 → stage2_label=0
- 正面 → stage2_label=1
"""
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from config import (
    MODEL_NAME, MAX_LEN, BATCH_SIZE, EVAL_BATCH_SIZE,
    EPOCHS, LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS,
    PROCESSED_DIR, STAGE2_MODEL_DIR,
)
from model import BertBinaryClassifier
from trainer_utils import SentimentDataset, load_data_for_stage, train_stage


def main():
    print("=" * 60)
    print("Stage2 训练：负面 vs 正面")
    print("=" * 60)

    # 1. 加载数据（仅非中性样本）
    train_texts, train_labels = load_data_for_stage(f"{PROCESSED_DIR}/stage2_train.csv")
    val_texts, val_labels = load_data_for_stage(f"{PROCESSED_DIR}/stage2_val.csv")

    print(f"训练集: {len(train_texts)} 条")
    print(f"  负面: {len(train_labels) - sum(train_labels)} | 正面: {sum(train_labels)}")
    print(f"验证集: {len(val_texts)} 条")
    print(f"  负面: {len(val_labels) - sum(val_labels)} | 正面: {sum(val_labels)}")

    # 2. Tokenizer & Dataset
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # 3. 模型
    config = BertConfig.from_pretrained(MODEL_NAME)
    model = BertBinaryClassifier(config)

    # 4. 训练
    model, history, best_f1 = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        stage_name="Stage2",
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_dir=STAGE2_MODEL_DIR,
    )

    print(f"\nStage2 训练完成！最佳 Val F1: {best_f1:.4f}")
    print(f"模型保存至: {STAGE2_MODEL_DIR}")


if __name__ == "__main__":
    main()
