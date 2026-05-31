"""
交互式推理脚本：用户输入文本，模型实时给出三分类情感预测及概率。
输入 "quit" / "exit" / "q" 退出。
"""
import torch
from transformers import BertTokenizer

from config import (
    MODEL_NAME, MAX_LEN, STAGE1_MODEL_DIR, STAGE2_MODEL_DIR, LABEL_NAMES,
)
from model import BertBinaryClassifier, CascadeSentimentModel


def load_model(device):
    """加载级联模型"""
    print("加载模型中...")
    stage1 = BertBinaryClassifier.from_pretrained(STAGE1_MODEL_DIR)
    stage2 = BertBinaryClassifier.from_pretrained(STAGE2_MODEL_DIR)
    cascade = CascadeSentimentModel(stage1, stage2)
    cascade.to(device)
    cascade.eval()
    tokenizer = BertTokenizer.from_pretrained(STAGE1_MODEL_DIR)
    print("模型加载完成！\n")
    return cascade, tokenizer


def predict(text: str, cascade, tokenizer, device):
    """单条预测"""
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }
    result = cascade.predict_with_probs(inputs, threshold=0.5)
    return result["predictions"][0], result["probs"][0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    cascade, tokenizer = load_model(device)

    print("=" * 60)
    print("  酒店评论情感分析 - 交互式推理")
    print("  输入 quit / exit / q 退出")
    print("=" * 60)
    print()

    while True:
        text = input("请输入评论: ").strip()

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        pred, probs = predict(text, cascade, tokenizer, device)
        label = LABEL_NAMES[pred]


        print(f"  预测结果: {label}")
        print(f"  概率分布: 负面 {probs[0]:.4f}  |  中性 {probs[1]:.4f}  |  正面 {probs[2]:.4f}")
        print()


if __name__ == "__main__":
    main()
