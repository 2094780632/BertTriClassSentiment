"""
酒店评论情感分析 Flask Web 应用
提供：单条分析、批量预测（上传 CSV/TXT）
"""
import sys
import os
import io
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer
from scripts.config import MAX_LEN, STAGE1_MODEL_DIR, STAGE2_MODEL_DIR, LABEL_NAMES
from scripts.model import BertBinaryClassifier, CascadeSentimentModel

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 最大 50MB 上传

# ---------- 加载模型 ----------

print("正在加载模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stage1 = BertBinaryClassifier.from_pretrained(STAGE1_MODEL_DIR)
stage2 = BertBinaryClassifier.from_pretrained(STAGE2_MODEL_DIR)
cascade = CascadeSentimentModel(stage1, stage2)
cascade.to(device)
cascade.eval()
tokenizer = BertTokenizer.from_pretrained(STAGE1_MODEL_DIR)
print(f"模型加载完成 | 设备: {device}")


def _predict_single(text):
    """核心推理函数，返回 (pred, probs_list)"""
    encoding = tokenizer(
        text, max_length=MAX_LEN, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    inputs = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }
    result = cascade.predict_with_probs(inputs, threshold=0.5)
    return result["predictions"][0], result["probs"][0]


# ---------- 路由 ----------

@app.route("/")
def index():
    return render_template("index.html", active_tab="single")


@app.route("/batch")
def batch():
    return render_template("index.html", active_tab="batch")


@app.route("/monitor")
def monitor():
    return render_template("index.html", active_tab="monitor")


@app.route("/api/predict", methods=["POST"])
def predict():
    """单条预测 API"""
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "请输入文本"}), 400

    pred, probs = _predict_single(text)
    return jsonify({
        "label": pred,
        "label_name": LABEL_NAMES[pred],
        "probs": {
            "negative": round(probs[0], 4),
            "neutral": round(probs[1], 4),
            "positive": round(probs[2], 4),
        }
    })


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """批量预测 API：上传 CSV/TXT 文件，逐行分析返回结果"""
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "请选择文件"}), 400

    filename = file.filename.lower()
    try:
        content = file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify({"error": "文件编码不支持，请使用 UTF-8 编码的 CSV/TXT 文件"}), 400

    lines = []
    if filename.endswith(".csv"):
        reader = csv.reader(io.StringIO(content))
        header = next(reader, None)
        text_col = 0
        if header:
            for ci, col in enumerate(header):
                if any(kw in str(col).lower() for kw in ["review", "text", "评论", "文本", "内容"]):
                    text_col = ci
                    break
        for row in reader:
            if row and len(row) > text_col:
                text = row[text_col].strip()
                if text:
                    lines.append(text)
    else:
        lines = [l.strip() for l in content.split("\n") if l.strip()]

    if not lines:
        return jsonify({"error": "文件中没有可分析的文本"}), 400

    results = []
    for idx, text in enumerate(lines):
        pred, probs = _predict_single(text)
        results.append({
            "index": idx + 1,
            "text": text[:200],
            "label": pred,
            "label_name": LABEL_NAMES[pred],
            "probs": {
                "negative": round(probs[0], 4),
                "neutral": round(probs[1], 4),
                "positive": round(probs[2], 4),
            }
        })

    # 统计
    total = len(results)
    neg = sum(1 for r in results if r["label"] == 0)
    neu = sum(1 for r in results if r["label"] == 1)
    pos = sum(1 for r in results if r["label"] == 2)

    return jsonify({
        "total": total,
        "negative_count": neg,
        "neutral_count": neu,
        "positive_count": pos,
        "results": results,
        "wordcloud": _generate_wordcloud(lines),  # 词云数据
    })


def _generate_wordcloud(texts, top_n=60):
    """从文本列表中统计词频，返回词云数据 [{text, size}, ...]"""
    from collections import Counter
    import re

    # 停用词
    stop = {"的","了","在","是","我","有","和","就","不","人","都","一","一个","上",
            "也","很","到","说","要","去","你","会","着","没","看","好","自己","这",
            "他","她","它","们","那","吧","吗","啊","呢","哦","嗯","但","还","可以",
            "这个","那个","什么","怎么","哪","为什么","觉得","感觉","有点","比较",
            "挺","非常","特别","真的","还是","所以","因为","如果","虽然","只是",
            "只有","就是","不是","没有","这些","那些","酒店","宾馆","房间","住","入住",
            "住过","住宿","服务","前台","早餐","位置","交通","设施","环境","价格"}

    all_words = []
    for text in texts:
        # 用 jieba 分词（如果可用），否则简单的按字符切分
        try:
            import jieba
            words = [w for w in jieba.cut(text) if len(w) >= 2 and w not in stop]
        except ImportError:
            words = [text[i:i+2] for i in range(len(text)-1)]
            words = [w for w in words if w not in stop]
        all_words.extend(words)

    counter = Counter(all_words)
    top_words = counter.most_common(top_n)

    if not top_words:
        return []

    max_count = top_words[0][1]
    min_count = top_words[-1][1]
    # 映射到 12-48 之间的字号
    if max_count == min_count:
        min_count = max_count - 1
    result = [
        {"text": w, "size": int(12 + (c - min_count) / (max_count - min_count) * 36)}
        for w, c in top_words
    ]
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
