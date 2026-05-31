"""
情感二分类模型训练脚本（融合版 - 另存为新模型）
=================================================
融合 ChnSentiCorp_htl_all（酒店）+ waimai_10k（外卖）数据
保存为独立的新模型，不覆盖旧模型

使用方法：
    python scripts/train_sentiment_fusion_v2.py

输出：
    model/sentiment_model_v2/  （新文件夹，不覆盖 sentiment_model/）
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import os
from pathlib import Path

# 设置镜像（国内加速下载）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============ 项目路径配置 ============
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# 新模型输出路径（不覆盖旧模型）
NEW_MODEL_DIR = BASE_DIR / 'model' / 'sentiment_model_v2'
OLD_MODEL_DIR = BASE_DIR / 'model' / 'sentiment_model'

# 确保输出目录存在
NEW_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("情感二分类模型 - 融合训练（另存为新模型）")
print("="*60)
print(f"\n旧模型路径: {OLD_MODEL_DIR}")
print(f"新模型路径: {NEW_MODEL_DIR}")
if OLD_MODEL_DIR.exists():
    print("  -> 旧模型存在，不会被覆盖")
else:
    print("  -> 旧模型不存在")

# ==================== 1. 加载数据 ====================
print("\n" + "="*60)
print("加载数据")
print("="*60)

# 加载酒店数据
htl_df = pd.read_csv(DATA_DIR / 'ChnSentiCorp_htl_all.csv')
htl_df['source'] = 'hotel'
print(f"酒店数据: {len(htl_df)} 条 (正面={sum(htl_df['label']==1)}, 负面={sum(htl_df['label']==0)})")

# 加载外卖数据
waimai_df = pd.read_csv(DATA_DIR / 'waimai_10k.csv')
waimai_df['source'] = 'waimai'
print(f"外卖数据: {len(waimai_df)} 条 (正面={sum(waimai_df['label']==1)}, 负面={sum(waimai_df['label']==0)})")

# 合并数据
df = pd.concat([htl_df, waimai_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n融合后总样本: {len(df)} 条")
print(f"  正面(1): {sum(df['label']==1)}")
print(f"  负面(0): {sum(df['label']==0)}")
print(f"  来源分布: 酒店={sum(df['source']=='hotel')}, 外卖={sum(df['source']=='waimai')}")

# 确保列名正确
df['label'] = df['label'].astype(int)
df['review'] = df['review'].astype(str).fillna('')

# ==================== 2. 划分数据集 ====================
print("\n" + "="*60)
print("划分数据集")
print("="*60)

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['review'].tolist(), df['label'].tolist(),
    test_size=0.2, random_state=42, stratify=df['label']
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels,
    test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"训练集: {len(train_texts)} 条")
print(f"验证集: {len(val_texts)} 条")
print(f"测试集: {len(test_texts)} 条")

# 确保类型正确
train_labels = [int(l) for l in train_labels]
val_labels   = [int(l) for l in val_labels]
test_labels  = [int(l) for l in test_labels]

train_texts = [str(t) for t in train_texts]
val_texts   = [str(t) for t in val_texts]
test_texts  = [str(t) for t in test_texts]

# ==================== 3. 加载Tokenizer和模型 ====================
print("\n" + "="*60)
print("加载预训练模型")
print("="*60)

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

print(f"模型: {model_name}")
print(f"分类数: 2 (正面/负面)")

# ==================== 4. Tokenization ====================
print("\n" + "="*60)
print("数据预处理（Tokenization）")
print("="*60)

def tokenize_function(examples):
    return tokenizer(
        examples['review'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

train_dataset = Dataset.from_dict({'review': train_texts, 'label': train_labels})
val_dataset   = Dataset.from_dict({'review': val_texts,   'label': val_labels})
test_dataset  = Dataset.from_dict({'review': test_texts,  'label': test_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print("Tokenization 完成")

# ==================== 5. 定义评估指标 ====================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ==================== 6. 设置训练参数 ====================
print("\n" + "="*60)
print("训练配置")
print("="*60)

training_args = TrainingArguments(
    output_dir=str(BASE_DIR / 'results_v2'),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=str(BASE_DIR / 'logs_v2'),
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

print(f"训练轮数: {training_args.num_train_epochs}")
print(f"批次大小: {training_args.per_device_train_batch_size}")
print(f"评估策略: 每 epoch 评估")
print(f"最佳模型标准: F1")

# ==================== 7. 创建 Trainer 并训练 ====================
print("\n" + "="*60)
print("开始训练")
print("="*60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==================== 8. 在测试集上评估 ====================
print("\n" + "="*60)
print("测试集评估")
print("="*60)

test_results = trainer.evaluate(test_dataset)
print(f"测试集结果:")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# ==================== 9. 保存为新模型 ====================
print("\n" + "="*60)
print("保存新模型")
print("="*60)

model.save_pretrained(NEW_MODEL_DIR)
tokenizer.save_pretrained(NEW_MODEL_DIR)

print(f"新模型已保存到: {NEW_MODEL_DIR}")
print(f"文件列表:")
for f in sorted(NEW_MODEL_DIR.iterdir()):
    size = f.stat().st_size / (1024*1024)  # MB
    print(f"  {f.name} ({size:.1f} MB)")

print("\n" + "="*60)
print("训练完成！")
print("="*60)
print(f"\n新模型路径: {NEW_MODEL_DIR}")
print(f"旧模型路径: {OLD_MODEL_DIR} (未被覆盖)")
print(f"\n切换模型使用:")
print(f"  方法1: 修改 predict.py 中的 SENTIMENT_MODEL_DIR")
print(f"  方法2: 重命名文件夹")
print(f"    mv model/sentiment_model model/sentiment_model_old")
print(f"    mv model/sentiment_model_v2 model/sentiment_model")