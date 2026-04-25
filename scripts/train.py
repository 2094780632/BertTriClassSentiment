"""
训练脚本：训练中性识别模块 + 情感二分类模型
===========================================

使用方法：
    python scripts/train.py

输出：
    model/model_neutral.bin
    model/model_sentiment.bin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path

# ============ 项目路径配置 ============
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    DATA_DIR = BASE_DIR / 'data'
    NEUTRAL_DATA = DATA_DIR / 'neutral_train_strict.csv'
    WAIMAI_DATA = DATA_DIR / 'waimai_10k.csv'
    HTL_DATA = DATA_DIR / 'ChnSentiCorp_htl_all.csv'

    MODEL_DIR = BASE_DIR / 'model'
    NEUTRAL_MODEL_PATH = MODEL_DIR / 'model_neutral.bin'
    SENTIMENT_MODEL_PATH = MODEL_DIR / 'model_sentiment.bin'

    PRETRAINED_MODEL = 'bert-base-chinese'
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 2e-5
    WARMUP_STEPS = 100

# 确保目录存在
Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============ 数据集类（修复版） ============
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

        # 修复：使用 tokenizer() 替代 encode_plus
        encoding = self.tokenizer(
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
        # 修复：使用 torch.optim.AdamW 替代 transformers 的 AdamW
        optimizer = AdamW(self.model.parameters(), lr=Config.LR, weight_decay=0.01)
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.WARMUP_STEPS,
            num_training_steps=total_steps
        )

        best_f1 = 0
        for epoch in range(Config.EPOCHS):
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

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        f1 = f1_score(true_labels, predictions, average='macro')
        report = classification_report(true_labels, predictions, digits=4)
        return f1, report

# ============ 训练中性识别模块 ============
def train_neutral_detector():
    print("="*60)
    print("训练中性识别模块")
    print("="*60)

    # 加载数据
    neutral_df = pd.read_csv(Config.NEUTRAL_DATA)
    neutral_df['label'] = 1

    waimai_df = pd.read_csv(Config.WAIMAI_DATA)
    htl_df = pd.read_csv(Config.HTL_DATA)
    non_neutral = pd.concat([waimai_df, htl_df])
    non_neutral['label'] = 0

    # 采样非中性数据（3倍中性数据量）
    n_neutral = len(neutral_df)
    non_neutral_sample = non_neutral.sample(n=min(n_neutral * 3, len(non_neutral)), random_state=42)

    # 合并数据
    train_df = pd.concat([
        neutral_df[['review', 'label']],
        non_neutral_sample[['review', 'label']]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"训练数据: 中性={n_neutral}, 非中性={len(non_neutral_sample)}")

    # 划分训练集和验证集
    train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # 创建数据集和数据加载器
    tokenizer = BertTokenizer.from_pretrained(Config.PRETRAINED_MODEL)
    train_dataset = SentimentDataset(train['review'].values, train['label'].values, tokenizer, Config.MAX_LEN)
    val_dataset = SentimentDataset(val['review'].values, val['label'].values, tokenizer, Config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # 训练
    trainer = ModelTrainer(num_labels=2, model_path=Config.NEUTRAL_MODEL_PATH)
    trainer.train(train_loader, val_loader)

    return trainer

# ============ 训练情感二分类模型 ============
def train_sentiment_model():
    print("\n" + "="*60)
    print("训练情感二分类模型")
    print("="*60)

    # 加载数据
    waimai_df = pd.read_csv(Config.WAIMAI_DATA)
    htl_df = pd.read_csv(Config.HTL_DATA)
    train_df = pd.concat([waimai_df, htl_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"训练数据: 正面={sum(train_df['label']==1)}, 负面={sum(train_df['label']==0)}")

    # 划分训练集和验证集
    train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # 创建数据集和数据加载器
    tokenizer = BertTokenizer.from_pretrained(Config.PRETRAINED_MODEL)
    train_dataset = SentimentDataset(train['review'].values, train['label'].values, tokenizer, Config.MAX_LEN)
    val_dataset = SentimentDataset(val['review'].values, val['label'].values, tokenizer, Config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # 训练
    trainer = ModelTrainer(num_labels=2, model_path=Config.SENTIMENT_MODEL_PATH)
    trainer.train(train_loader, val_loader)

    return trainer

# ============ 主程序 ============
if __name__ == '__main__':
    print("开始训练...")
    print(f"项目根目录: {BASE_DIR}")
    print(f"数据目录: {Config.DATA_DIR}")
    print(f"模型输出目录: {Config.MODEL_DIR}")

    # 检查数据文件是否存在
    for f in [Config.NEUTRAL_DATA, Config.WAIMAI_DATA, Config.HTL_DATA]:
        if not f.exists():
            print(f"错误: 找不到数据文件 {f}")
            exit(1)

    # 训练
    neutral_trainer = train_neutral_detector()
    sentiment_trainer = train_sentiment_model()

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型已保存到:")
    print(f"  {Config.NEUTRAL_MODEL_PATH}")
    print(f"  {Config.SENTIMENT_MODEL_PATH}")
    print("\n现在可以运行 predict.py 进行预测")