"""
酒店评论情感分析客户端 - PyQt5
基于 form.ui 构建界面，三个标签页：
  Tab 0: 单条分析 - 输入文本，输出三分类概率
  Tab 1: 批量预测 - 占位
  Tab 2: 监控分析 - 占位
"""
import os
import sys
import torch
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QGroupBox, QProgressBar, QWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from transformers import BertTokenizer

# 将项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.config import MAX_LEN, STAGE1_MODEL_DIR, STAGE2_MODEL_DIR, LABEL_NAMES
from scripts.model import BertBinaryClassifier, CascadeSentimentModel


# ========== 后台推理线程 ==========

class PredictWorker(QThread):
    finished = pyqtSignal(int, list)

    def __init__(self, cascade, tokenizer, device, text):
        super().__init__()
        self.cascade = cascade
        self.tokenizer = tokenizer
        self.device = device
        self.text = text

    def run(self):
        encoding = self.tokenizer(
            self.text, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        inputs = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }
        result = self.cascade.predict_with_probs(inputs, threshold=0.5)
        self.finished.emit(result["predictions"][0], result["probs"][0])


# ========== 主窗口 ==========

class SentimentClient(QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. 加载 form.ui
        ui_path = os.path.join(os.path.dirname(__file__), "form.ui")
        uic.loadUi(ui_path, self)

        # 2. 加载模型
        self._init_model()

        # 3. 往 form.ui 的 tab 里填充内容
        self._init_tab_single()     # Tab 0: 单条分析 (ui 中 name="tab")
        self._init_tab_batch()      # Tab 1: 批量预测 (ui 中 name="tab_2")
        self._init_tab_monitor()    # Tab 2: 监控分析 (ui 中 name="tab_3")

        self.worker = None

    # ---------- 模型 ----------

    def _init_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.statusbar.showMessage(f"正在加载模型... 设备: {self.device}")

        stage1 = BertBinaryClassifier.from_pretrained(STAGE1_MODEL_DIR)
        stage2 = BertBinaryClassifier.from_pretrained(STAGE2_MODEL_DIR)
        self.cascade = CascadeSentimentModel(stage1, stage2)
        self.cascade.to(self.device)
        self.cascade.eval()
        self.tokenizer = BertTokenizer.from_pretrained(STAGE1_MODEL_DIR)

        self.statusbar.showMessage(f"✅ 模型加载完成 | 设备: {self.device}")

    # ---------- Tab 0: 单条分析 ----------

    def _init_tab_single(self):
        tab = self.tabWidget.widget(0)  # form.ui 中的 name="tab"
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # 输入区
        input_group = QGroupBox("输入评论")
        input_group.setFont(QFont("等线", 11, QFont.Bold))
        input_layout = QVBoxLayout(input_group)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "请输入酒店评论文本，例如：酒店环境不错，服务也很热情，下次还会再来。"
        )
        self.text_input.setMaximumHeight(100)
        self.text_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #d0d0d0; border-radius: 4px;
                padding: 6px; font-size: 20px;
            }
            QTextEdit:focus { border: 1px solid #1890ff; }
        """)
        input_layout.addWidget(self.text_input)
        layout.addWidget(input_group)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.submit_btn = QPushButton("🔍 开始分析")
        self.submit_btn.setMinimumSize(130, 38)
        self.submit_btn.setStyleSheet(self._btn_css("#1890ff", "#40a9ff"))
        self.submit_btn.clicked.connect(self._on_predict)
        btn_row.addWidget(self.submit_btn)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.setMinimumSize(70, 38)
        self.clear_btn.setStyleSheet(self._btn_css("#8c8c8c", "#bfbfbf"))
        self.clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(3)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)

        # 结果区
        result_group = QGroupBox("分析结果")
        result_group.setFont(QFont("等线", 11, QFont.Bold))
        result_layout = QVBoxLayout(result_group)

        self.result_text = QLabel("等待输入...")
        self.result_text.setFont(QFont("等线", 18, QFont.Bold))
        self.result_text.setAlignment(Qt.AlignCenter)
        self.result_text.setStyleSheet("color: #333; padding: 6px;")
        result_layout.addWidget(self.result_text)

        # 三条概率条
        prob_box = QWidget()
        prob_box.setStyleSheet("background: #fafafa; border-radius: 6px; padding: 6px;")
        prob_layout = QVBoxLayout(prob_box)
        self.neg_bar_row = self._make_bar("负面", "#ff4d4f")
        self.neu_bar_row = self._make_bar("中性", "#faad14")
        self.pos_bar_row = self._make_bar("正面", "#52c41a")
        prob_layout.addWidget(self.neg_bar_row)
        prob_layout.addWidget(self.neu_bar_row)
        prob_layout.addWidget(self.pos_bar_row)
        result_layout.addWidget(prob_box)

        layout.addWidget(result_group)

    def _make_bar(self, name, color):
        row = QWidget()
        l = QHBoxLayout(row)
        l.setContentsMargins(0, 2, 0, 2)
        label = QLabel(name)
        label.setFixedWidth(40)
        label.setFont(QFont("等线", 11, QFont.Bold))
        label.setStyleSheet(f"color: {color};")
        l.addWidget(label)
        bar = QProgressBar()
        bar.setRange(0, 10000)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setStyleSheet(f"""
            QProgressBar {{
                border: none; background: #e8e8e8; border-radius: 4px;
                height: 28px; text-align: right; font-size: 14px; padding-right: 6px;
            }}
            QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}
        """)
        l.addWidget(bar)
        return row

    def _on_predict(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.statusbar.showMessage("⚠️ 请输入文本")
            return
        self.submit_btn.setEnabled(False)
        self.progress.show()
        self.statusbar.showMessage("正在分析...")
        self.worker = PredictWorker(self.cascade, self.tokenizer, self.device, text)
        self.worker.finished.connect(self._on_result)
        self.worker.start()

    def _on_result(self, pred, probs):
        self.submit_btn.setEnabled(True)
        self.progress.hide()
        emoji = {0: "🔴", 1: "🟡", 2: "🟢"}
        self.result_text.setText(f"{emoji[pred]}  {LABEL_NAMES[pred]}")
        for row, val in [(self.neg_bar_row, probs[0]),
                          (self.neu_bar_row, probs[1]),
                          (self.pos_bar_row, probs[2])]:
            bar = row.layout().itemAt(1).widget()
            bar.setValue(int(val * 10000))
            bar.setFormat(f"{val:.2%}")
        self.statusbar.showMessage(f"✅ 分析完成 → {LABEL_NAMES[pred]}")

    def _on_clear(self):
        self.text_input.clear()
        self.result_text.setText("等待输入...")
        for row in (self.neg_bar_row, self.neu_bar_row, self.pos_bar_row):
            bar = row.layout().itemAt(1).widget()
            bar.setValue(0)
            bar.setFormat("")

    # ---------- Tab 1 & 2: 占位 ----------

    def _init_tab_batch(self):
        self._placeholder(self.tabWidget.widget(1), "📂 批量预测",
                          "批量上传 CSV / Excel 进行预测，支持结果导出。\n\n此功能即将上线，敬请期待。")

    def _init_tab_monitor(self):
        self._placeholder(self.tabWidget.widget(2), "📊 监控分析",
                          "实时监控分析请求量、响应时间、情感分布等指标。\n\n此功能即将上线，敬请期待。")

    def _placeholder(self, parent_widget, title, desc):
        inner = QWidget()
        lo = QVBoxLayout(inner)
        lo.setAlignment(Qt.AlignCenter)
        icon = QLabel("🚧")
        icon.setFont(QFont("Segoe UI Emoji", 48))
        icon.setAlignment(Qt.AlignCenter)
        lo.addWidget(icon)
        t = QLabel(title)
        t.setFont(QFont("等线", 16, QFont.Bold))
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("color: #555; margin-top: 8px;")
        lo.addWidget(t)
        d = QLabel(desc)
        d.setFont(QFont("等线", 11))
        d.setAlignment(Qt.AlignCenter)
        d.setWordWrap(True)
        d.setStyleSheet("color: #999; margin-top: 12px;")
        lo.addWidget(d)

        parent_layout = parent_widget.layout()
        if parent_layout is None:
            parent_layout = QVBoxLayout(parent_widget)
        parent_layout.addWidget(inner)

    # ---------- 样式工具 ----------

    @staticmethod
    def _btn_css(bg, hover):
        return f"""
            QPushButton {{
                background: {bg}; color: white; border: none;
                border-radius: 4px; font-size: 15px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ background: {bg}; }}
            QPushButton:disabled {{ background: #d9d9d9; }}
        """


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("等线", 10))
    window = SentimentClient()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
