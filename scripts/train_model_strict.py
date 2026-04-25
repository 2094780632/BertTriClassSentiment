
"""
中文评论情感三分类模型训练脚本（方案A：保守策略）
=====================================================
项目结构：
  /
  ├── data/
  │   ├── neutral_train_strict.csv
  │   ├── waimai_10k.csv
  │   └── ChnSentiCorp_htl_all.csv
  ├── model/
  │   ├── model_neutral.bin
  │   └── model_sentiment.bin
  └── scripts/
      └── train_model.py  (本文件)

使用严格筛选的中性数据（得分>=0.8，共1692条）

级联策略：
  1. 中性识别模块（二分类：中性 vs 非中性）
  2. 情感二分类模型（正面 vs 负面）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
import os
from tqdm import tqdm
from pathlib import Path

# ============ 项目路径配置 ============
# 获取项目根目录（scripts的父目录）
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # 数据路径（相对于项目根目录）
    DATA_DIR = BASE_DIR / 'data'
    NEUTRAL_DATA = DATA_DIR / 'neutral_train_strict.csv'   # 严格中性数据
    WAIMAI_DATA = DATA_DIR / 'waimai_10k.csv'              # 外卖数据
    HTL_DATA = DATA_DIR / 'ChnSentiCorp_htl_all.csv'                    # 酒店数据

    # 模型输出路径
    MODEL_DIR = BASE_DIR / 'model'
    NEUTRAL_MODEL_PATH = MODEL_DIR / 'model_neutral.bin'
    SENTIMENT_MODEL_PATH = MODEL_DIR / 'model_sentiment.bin'

    # 模型配置
    PRETRAINED_MODEL = 'bert-base-chinese'
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 2e-5
    WARMUP_STEPS = 100

    # 中性判定阈值
    NEUTRAL_THRESHOLD = 0.6

# 确保目录存在
Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============ 数据集类 ============
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============ 模型训练器 ============
class ModelTrainer:
    def __init__(self, num_labels, model_path, pretrained_model=Config.PRETRAINED_MODEL):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels
        ).to(self.device)
        self.model_path = model_path

    def train(self, train_loader, val_loader):
        optimizer = AdamW(self.model.parameters(), lr=Config.LR, weight_decay=0.01)
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.WARMUP_STEPS,
            num_training_steps=total_steps
        )

        best_f1 = 0
        for epoch in range(Config.EPOCHS):
            # 训练
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS}')

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)

            # 验证
            val_f1, val_report = self.evaluate(val_loader)
            print(f'\nEpoch {epoch+1} - Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}')
            print(val_report)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), self.model_path)
                print(f'  -> Saved best model to {self.model_path} (F1={best_f1:.4f})')

    def evaluate(self, data_loader):
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = torch.argmax(outputs.logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        f1 = f1_score(true_labels, predictions, average='macro')
        report = classification_report(true_labels, predictions, digits=4)
        return f1, report

# ============ 1. 训练中性识别模块 ============
def train_neutral_detector():
    """
    训练中性识别模块（二分类：中性 vs 非中性）
    使用严格中性数据（1692条）
    """
    print("="*60)
    print("训练中性识别模块（严格数据）")
    print("="*60)

    # 加载中性数据
    neutral_df = pd.read_csv(Config.NEUTRAL_DATA)
    neutral_df['label'] = 1  # 1 = 中性

    # 加载非中性数据
    waimai_df = pd.read_csv(Config.WAIMAI_DATA)
    htl_df = pd.read_csv(Config.HTL_DATA)
    non_neutral = pd.concat([waimai_df, htl_df])
    non_neutral['label'] = 0  # 0 = 非中性

    # 平衡数据：非中性 = 中性 * 3
    n_neutral = len(neutral_df)
    non_neutral_sample = non_neutral.sample(n=min(n_neutral * 3, len(non_neutral)), random_state=42)

    # 合并
    train_df = pd.concat([
        neutral_df[['Review', 'label']],
        non_neutral_sample[['Review', 'label']]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"训练数据: 中性={n_neutral}, 非中性={len(non_neutral_sample)}")

    # 划分
    train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # 创建DataLoader
    train_dataset = SentimentDataset(train['Review'].values, train['label'].values, 
                                      BertTokenizer.from_pretrained(Config.PRETRAINED_MODEL), Config.MAX_LEN)
    val_dataset = SentimentDataset(val['Review'].values, val['label'].values,
                                    BertTokenizer.from_pretrained(Config.PRETRAINED_MODEL), Config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # 训练
    trainer = ModelTrainer(num_labels=2, model_path=Config.NEUTRAL_MODEL_PATH)
    trainer.train(train_loader, val_loader)

    return trainer

# ============ 2. 训练情感二分类模型 ============
def train_sentiment_model():
    """
    训练情感二分类模型（正面 vs 负面）
    """
    print("\n" + "="*60)
    print("训练情感二分类模型")
    print("="*60)

    # 加载数据
    waimai_df = pd.read_csv(Config.WAIMAI_DATA)
    htl_df = pd.read_csv(Config.HTL_DATA)
    train_df = pd.concat([waimai_df, htl_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"训练数据: 正面={sum(train_df['label']==1)}, 负面={sum(train_df['label']==0)}")

    # 划分
    train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # 创建DataLoader
    train_dataset = SentimentDataset(train['Review'].values, train['label'].values,
                                      BertTokenizer.from_pretrained(Config.PRETRAINED_MODEL), Config.MAX_LEN)
    val_dataset = SentimentDataset(val['Review'].values, val['label'].values,
                                    BertTokenizer.from_pretrained(Config.PRETRAINED_MODEL), Config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # 训练
    trainer = ModelTrainer(num_labels=2, model_path=Config.SENTIMENT_MODEL_PATH)
    trainer.train(train_loader, val_loader)

    return trainer

# ============ 3. 级联预测器 ============
class CascadePredictor:
    """
    级联预测器
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(Config.PRETRAINED_MODEL)

        # 加载中性识别模型
        self.neutral_model = BertForSequenceClassification.from_pretrained(
            Config.PRETRAINED_MODEL, num_labels=2
        ).to(self.device)
        self.neutral_model.load_state_dict(
            torch.load(Config.NEUTRAL_MODEL_PATH, map_location=self.device)
        )
        self.neutral_model.eval()

        # 加载情感二分类模型
        self.sentiment_model = BertForSequenceClassification.from_pretrained(
            Config.PRETRAINED_MODEL, num_labels=2
        ).to(self.device)
        self.sentiment_model.load_state_dict(
            torch.load(Config.SENTIMENT_MODEL_PATH, map_location=self.device)
        )
        self.sentiment_model.eval()

    def predict(self, text, neutral_threshold=None):
        """
        预测单条文本

        Args:
            text: 输入文本
            neutral_threshold: 中性阈值（默认使用Config中的值）

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

        encoding = self.tokenizer.encode_plus(
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

        # 第一步：中性识别
        with torch.no_grad():
            outputs = self.neutral_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            neutral_prob = probs[0][1].item()
            non_neutral_prob = probs[0][0].item()

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

        # 第二步：情感二分类
        with torch.no_grad():
            outputs = self.sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            neg_prob = probs[0][0].item()
            pos_prob = probs[0][1].item()

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
    # 训练两个模型（取消注释以执行）
    # neutral_trainer = train_neutral_detector()
    # sentiment_trainer = train_sentiment_model()

    # 使用级联预测器
    print("\n加载预训练模型...")
    predictor = CascadePredictor()

    # 测试
    test_texts = [
        "味道很好，下次还会来！",           # 正面
        "太难吃了，投诉！",                 # 负面
        "味道还行，就是有点贵",             # 中性
        "一般般，无功无过",                 # 中性
        "送餐很快，但是菜凉了",             # 中性
        "还可以，就是速度慢了点",             # 中性
        "非常不错，强烈推荐！",             # 正面
        "太差劲了，再也不会来了",           # 负面
    ]

    print("\n" + "="*60)
    print("预测测试")
    print("="*60)
    for text in test_texts:
        result = predictor.predict(text)
        print(f"[{result['label_name']}] 置信度:{result['confidence']:.3f} | {text}")
        if result['label'] == 2:
            print(f"         中性概率: {result['details']['neutral_prob']:.3f}")
        else:
            print(f"         正/负概率: {result['details']['positive_prob']:.3f}/{result['details']['negative_prob']:.3f}")
