"""
预测脚本：加载训练好的模型进行情感三分类预测
===============================================

使用方法：
    python scripts/predict.py

依赖文件：
    model/
    ├── model_neutral.bin          <- 中性识别模型
    └── sentiment_model/           <- 情感二分类模型（可替换）
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── tokenizer_config.json

支持两种二分类模型格式：
    1. 单文件 .bin
    2. 文件夹（含 config.json + model.safetensors）
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path

# ============ 项目路径配置 ============
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    MODEL_DIR = BASE_DIR / 'model'

    # 中性识别模型
    NEUTRAL_MODEL_PATH = MODEL_DIR / 'model_neutral.bin'

    # 情感二分类模型（支持两种格式）
    SENTIMENT_MODEL_BIN = MODEL_DIR / 'model_sentiment.bin'
    SENTIMENT_MODEL_DIR = MODEL_DIR / 'sentiment_model_v2'

    # 预训练模型名称
    PRETRAINED_MODEL = 'bert-base-chinese'
    MAX_LEN = 128
    NEUTRAL_THRESHOLD = 0.6

# ============ 级联预测器 ============
class CascadePredictor:
    """
    级联预测器：
    1. 先用中性识别模块判断是否为中性
    2. 若非中性，再用二分类模型判断正/负
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # ========== 加载中性识别模块 ==========
        print("加载中性识别模型...")
        self.neutral_tokenizer = BertTokenizer.from_pretrained(Config.PRETRAINED_MODEL)
        self.neutral_model = BertForSequenceClassification.from_pretrained(
            Config.PRETRAINED_MODEL, num_labels=2
        ).to(self.device)
        self.neutral_model.load_state_dict(
            torch.load(Config.NEUTRAL_MODEL_PATH, map_location=self.device)
        )
        self.neutral_model.eval()

        # ========== 加载情感二分类模型 ==========
        print("加载情感二分类模型...")

        if Config.SENTIMENT_MODEL_DIR.exists() and (Config.SENTIMENT_MODEL_DIR / 'config.json').exists():
            # 格式: 文件夹
            print(f"  检测到文件夹格式: {Config.SENTIMENT_MODEL_DIR}")
            self.sentiment_tokenizer = BertTokenizer.from_pretrained(Config.SENTIMENT_MODEL_DIR)
            self.sentiment_model = BertForSequenceClassification.from_pretrained(
                Config.SENTIMENT_MODEL_DIR
            ).to(self.device)

        else:
            raise FileNotFoundError(
                f"找不到情感二分类模型！\n"
                f"请确保以下路径之一存在：\n"
                f"  1. {Config.SENTIMENT_MODEL_DIR} (文件夹)\n"
                #f"  2. {Config.SENTIMENT_MODEL_BIN} (单文件 .bin)"
            )

        self.sentiment_model.eval()
        print("模型加载完成！")

    def predict(self, text, neutral_threshold=None):
        """
        预测单条文本

        Args:
            text: 输入文本
            neutral_threshold: 中性阈值（默认0.6）

        Returns:
            dict: {
                'label': 0=负面, 1=正面, 2=中性,
                'label_name': '负面'/'正面'/'中性',
                'confidence': 置信度,
                'details': 详细概率
            }
        """
        if neutral_threshold is None:
            neutral_threshold = Config.NEUTRAL_THRESHOLD

        # ========== 第一步：中性识别 ==========
        encoding = self.neutral_tokenizer(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.neutral_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            neutral_prob = probs[0][1].item()
            non_neutral_prob = probs[0][0].item()

        # 判定为中性
        if neutral_prob >= neutral_threshold:
            return {
                'label': 2,
                'label_name': '中性',
                'confidence': neutral_prob,
                'details': {
                    'neutral_prob': neutral_prob,
                    'non_neutral_prob': non_neutral_prob
                }
            }

        # ========== 第二步：情感二分类 ==========
        # 使用二分类模型自己的 tokenizer
        encoding = self.sentiment_tokenizer(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            neg_prob = probs[0][0].item()
            pos_prob = probs[0][1].item()

        # 判定为正面或负面
        if pos_prob >= 0.5:
            return {
                'label': 1,
                'label_name': '正面',
                'confidence': pos_prob,
                'details': {
                    'positive_prob': pos_prob,
                    'negative_prob': neg_prob
                }
            }
        else:
            return {
                'label': 0,
                'label_name': '负面',
                'confidence': neg_prob,
                'details': {
                    'positive_prob': pos_prob,
                    'negative_prob': neg_prob
                }
            }

    def predict_batch(self, texts, neutral_threshold=None):
        """批量预测"""
        return [self.predict(t, neutral_threshold) for t in texts]

# ============ 主程序 ============
if __name__ == '__main__':
    # 检查中性模型文件
    if not Config.NEUTRAL_MODEL_PATH.exists():
        print(f"错误: 找不到中性识别模型 {Config.NEUTRAL_MODEL_PATH}")
        print("请先运行 train.py 训练模型")
        exit(1)

    # 加载预测器
    predictor = CascadePredictor()

    # 测试样例
    test_texts = [
        "味道很好，下次还会来！",
        "太难吃了，投诉！",
        "味道还行，就是有点贵",
        "一般般，无功无过",
        "送餐很快，但是菜凉了",
        "还可以，就是速度慢了点",
        "非常不错，强烈推荐！",
        "太差劲了，再也不会来了",
    ]

    print("\n" + "="*60)
    print("预测测试")
    print("="*60)

    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n[{result['label_name']}] 置信度:{result['confidence']:.3f}")
        print(f"  文本: {text}")

        if result['label'] == 2:
            print(f"  中性概率: {result['details']['neutral_prob']:.3f}")
        else:
            print(f"  正面概率: {result['details']['positive_prob']:.3f}")
            print(f"  负面概率: {result['details']['negative_prob']:.3f}")

    # 交互式预测
    print("\n" + "="*60)
    print("输入文本进行预测（输入 'quit' 退出）")
    print("="*60)

    while True:
        text = input("\n> ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        if not text:
            continue

        result = predictor.predict(text)
        print(f"[{result['label_name']}] 置信度:{result['confidence']:.3f} | {text}")