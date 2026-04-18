import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 1. 加载数据和模型 ====================
# 读取原始数据（与训练时相同）
df = pd.read_csv("ChnSentiCorp_htl_all.csv")
df['label'] = df['label'].astype(int)
df['review'] = df['review'].astype(str).fillna('')

# 重新划分测试集（使用相同的 random_state 以保证与训练时的测试集一致）
_, temp_texts, _, temp_labels = train_test_split(
    df['review'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)
_, test_texts, _, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# 类型转换
test_labels = [int(l) for l in test_labels]
test_texts = [str(t) for t in test_texts]

# 加载保存的模型和分词器
model_path = "./sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 创建测试集的 Dataset 对象并进行 tokenization
def tokenize_function(examples):
    return tokenizer(
        examples['review'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

test_dataset = Dataset.from_dict({'review': test_texts, 'label': test_labels})
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ==================== 2. 预测 ====================
# 使用 Trainer 进行预测（需要包装一下）
trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

# ==================== 3. 混淆矩阵 ====================
cm = confusion_matrix(test_labels, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
print("混淆矩阵已保存为 confusion_matrix.png")