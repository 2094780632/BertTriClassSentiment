from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载模型和分词器
model_path = "./sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# 测试句子
test_texts = [
    "这家酒店真的太棒了，服务很好，下次还会来！",  # 正面
    "房间又小又脏，设施陈旧，不会再住了。",      # 负面
    "位置还行，但价格偏贵。",                    # 中性（训练集无中性标签，模型会倾向某一类）
]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax().item()
    return "正面" if pred == 1 else "负面"

for text in test_texts:
    print(f"文本: {text}\n预测: {predict(text)}\n{'-'*40}")