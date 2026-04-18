import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# 设置镜像（国内加速下载）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 1. 加载数据 ====================
# 读取CSV文件（根据你的实际文件名修改）
df = pd.read_csv("ChnSentiCorp_htl_all.csv")

# 确保列名正确（你的文件有 label 和 review 两列）
# 检查数据：标签必须是整数，评论必须是字符串
df['label'] = df['label'].astype(int)          # 转换为整数
df['review'] = df['review'].astype(str).fillna('')  # 转换为字符串，缺失值填充空串

# 查看数据分布
print(f"数据集总样本数: {len(df)}")
print(f"正面(1)样本数: {len(df[df['label']==1])}")
print(f"负面(0)样本数: {len(df[df['label']==0])}")

# ==================== 2. 划分数据集 ====================
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['review'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}, 测试集大小: {len(test_texts)}")

# 确保标签是 Python int，文本是 Python str（避免 ArrowTypeError）
train_labels = [int(l) for l in train_labels]
val_labels   = [int(l) for l in val_labels]
test_labels  = [int(l) for l in test_labels]

train_texts = [str(t) for t in train_texts]
val_texts   = [str(t) for t in val_texts]
test_texts  = [str(t) for t in test_texts]

# ==================== 3. 加载Tokenizer和模型 ====================
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ==================== 4. Tokenization ====================
def tokenize_function(examples):
    return tokenizer(
        examples['review'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# 转换为Dataset对象
train_dataset = Dataset.from_dict({'review': train_texts, 'label': train_labels})
val_dataset   = Dataset.from_dict({'review': val_texts,   'label': val_labels})
test_dataset  = Dataset.from_dict({'review': test_texts,  'label': test_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

# 设置数据格式为PyTorch张量
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

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
training_args = TrainingArguments(
    output_dir='./results',          # 保存结果目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=16,  # 训练批次大小（如果显存不够，减小到8或4）
    per_device_eval_batch_size=64,   # 评估批次大小
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",     # 每个epoch后评估
    save_strategy="epoch",
    load_best_model_at_end=True,     # 训练结束后加载最佳模型
    metric_for_best_model="accuracy",
)

# ==================== 7. 创建Trainer并训练 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("开始训练...")
trainer.train()

# ==================== 8. 在测试集上评估 ====================
print("在测试集上评估...")
test_results = trainer.evaluate(test_dataset)
print(f"测试集结果: {test_results}")

# ==================== 9. 保存最终模型 ====================
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
print("模型已保存到 ./sentiment_model")