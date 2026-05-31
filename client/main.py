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
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
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
    """单条推理（强制 CPU，避免 CUDA 跨线程崩溃）"""
    finished = pyqtSignal(int, list)

    def __init__(self, cascade, tokenizer, device, text):
        super().__init__()
        self.cascade = cascade
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")   # 强制 CPU
        self.text = text

    def run(self):
        self.cascade.to("cpu")
        encoding = self.tokenizer(
            self.text, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        result = self.cascade.predict_with_probs(inputs, threshold=0.5)
        self.finished.emit(result["predictions"][0], result["probs"][0])


class BatchPredictWorker(QThread):
    """逐行流式批量推理：从文件逐行读取→判断→释放，避免内存爆炸"""
    progress = pyqtSignal(int)               # current index
    result_row = pyqtSignal(int, str, int, list)  # idx, text, pred, probs
    total_count = pyqtSignal(int)            # 总行数
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, cascade, tokenizer, file_path):
        super().__init__()
        self.cascade = cascade
        self.tokenizer = tokenizer
        self.file_path = file_path

    def run(self):
        self.cascade.to("cpu")
        idx = 0
        try:
            if self.file_path.endswith(".csv"):
                import csv
                with open(self.file_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    # 找文本列
                    text_col = 0
                    if header:
                        for ci, col in enumerate(header):
                            if any(kw in str(col) for kw in ["review", "text", "评论", "文本", "内容"]):
                                text_col = ci
                                break
                    # 先数总行数
                    all_rows = list(reader)
                    total = len(all_rows)
                    self.total_count.emit(total)
                    for row in all_rows:
                        if not row or all(c.strip() == "" for c in row):
                            self.progress.emit(idx + 1)
                            idx += 1
                            continue
                        text = row[text_col].strip()
                        if not text:
                            self.progress.emit(idx + 1)
                            idx += 1
                            continue
                        pred, probs = self._predict_one(text)
                        self.result_row.emit(idx, text, pred, probs)
                        self.progress.emit(idx + 1)
                        idx += 1
            else:
                # TXT / 其他：逐行读取
                with open(self.file_path, "r", encoding="utf-8-sig") as f:
                    lines = [l.strip() for l in f if l.strip()]
                total = len(lines)
                self.total_count.emit(total)
                for line in lines:
                    text = line.strip()
                    if not text:
                        self.progress.emit(idx + 1)
                        idx += 1
                        continue
                    pred, probs = self._predict_one(text)
                    self.result_row.emit(idx, text, pred, probs)
                    self.progress.emit(idx + 1)
                    idx += 1
        except Exception as e:
            self.error.emit(str(e))
            return
        self.finished.emit()

    def _predict_one(self, text):
        encoding = self.tokenizer(
            text, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        result = self.cascade.predict_with_probs(inputs, threshold=0.5)
        return result["predictions"][0], result["probs"][0]


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

    # ---------- Tab 1: 批量预测 ----------

    def _init_tab_batch(self):
        tab = self.tabWidget.widget(1)  # form.ui 中的 name="tab_2"
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # ---- 操作区 ----
        top_row = QHBoxLayout()

        self.batch_file_label = QLabel("未选择文件")
        self.batch_file_label.setFont(QFont("等线", 11))
        self.batch_file_label.setStyleSheet("color: #888; padding: 4px;")
        top_row.addWidget(self.batch_file_label)

        top_row.addStretch()

        select_btn = QPushButton("📁 选择文件")
        select_btn.setMinimumSize(110, 36)
        select_btn.setStyleSheet(self._btn_css("#722ed1", "#9254de"))
        select_btn.clicked.connect(self._on_select_file)
        top_row.addWidget(select_btn)

        self.batch_run_btn = QPushButton("🚀 开始分析")
        self.batch_run_btn.setMinimumSize(130, 36)
        self.batch_run_btn.setStyleSheet(self._btn_css("#1890ff", "#40a9ff"))
        self.batch_run_btn.clicked.connect(self._on_batch_predict)
        self.batch_run_btn.setEnabled(False)
        top_row.addWidget(self.batch_run_btn)

        layout.addLayout(top_row)

        # ---- 进度条 ----
        self.batch_progress = QProgressBar()
        self.batch_progress.setMaximumHeight(24)
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #d0d0d0; border-radius: 4px;
                text-align: center; font-size: 12px;
            }
            QProgressBar::chunk { background: #1890ff; border-radius: 3px; }
        """)
        self.batch_progress.setValue(0)
        layout.addWidget(self.batch_progress)

        # ---- 结果表格 ----
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(4)
        self.batch_table.setHorizontalHeaderLabels(["序号", "评论文本", "预测结果", "负面 / 中性 / 正面"])
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.batch_table.setColumnWidth(0, 50)
        self.batch_table.setColumnWidth(2, 100)
        self.batch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setStyleSheet("""
            QTableWidget {
                font-size: 12px; gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background: #f5f5f5; font-weight: bold; padding: 6px;
                border: none; border-bottom: 2px solid #d0d0d0;
            }
        """)
        layout.addWidget(self.batch_table)

        # ---- 统计摘要 ----
        self.batch_summary = QLabel("")
        self.batch_summary.setFont(QFont("等线", 12))
        self.batch_summary.setStyleSheet("""
            background: #fafafa; border-radius: 6px; padding: 10px; color: #333;
        """)
        self.batch_summary.setWordWrap(True)
        self.batch_summary.hide()
        layout.addWidget(self.batch_summary)

        # ---- 导出按钮 ----
        export_row = QHBoxLayout()
        export_row.addStretch()
        self.export_btn = QPushButton("💾 导出结果 CSV")
        self.export_btn.setMinimumSize(140, 36)
        self.export_btn.setStyleSheet(self._btn_css("#52c41a", "#73d13d"))
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        self.export_btn.hide()
        export_row.addWidget(self.export_btn)
        layout.addLayout(export_row)

        # 状态
        self.batch_file_path = ""
        self.batch_texts = []
        self.batch_results = []  # list of (pred, probs)

    def _on_select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "",
            "文本/表格文件 (*.csv *.txt);;所有文件 (*)"
        )
        if not path:
            return

        self.batch_file_path = path
        self.batch_file_label.setText(f"📄 {os.path.basename(path)}")
        self.batch_run_btn.setEnabled(True)
        self.batch_table.setRowCount(0)
        self.batch_summary.hide()
        self.export_btn.hide()
        self.export_btn.setEnabled(False)
        self.batch_progress.setValue(0)
        self.batch_results = []
        self.batch_texts = []
        self.statusbar.showMessage(f"✅ 已选择文件: {os.path.basename(path)}")

    def _on_batch_predict(self):
        file_path = getattr(self, "batch_file_path", "")
        if not file_path:
            return
        self.batch_results = []
        self.batch_texts = []
        self.batch_table.setRowCount(0)
        self.batch_run_btn.setEnabled(False)
        self.export_btn.hide()
        self.batch_summary.hide()
        self.batch_progress.setRange(0, 0)
        self.batch_progress.setValue(0)

        self.batch_worker = BatchPredictWorker(
            self.cascade, self.tokenizer, file_path,
        )
        self.batch_worker.total_count.connect(self._on_batch_total)
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.result_row.connect(self._on_batch_row)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.error.connect(self._on_batch_error)
        self.batch_worker.start()
        self.statusbar.showMessage("正在逐行分析...")

    def _on_batch_total(self, total):
        self.batch_progress.setRange(0, total)
        self.batch_table.setRowCount(total)

    def _on_batch_progress(self, current):
        self.batch_progress.setValue(current)

    def _on_batch_row(self, row_idx, text, pred, probs):
        emoji = {0: "🔴", 1: "🟡", 2: "🟢"}
        label = LABEL_NAMES[pred]
        prob_str = f"{probs[0]:.1%} / {probs[1]:.1%} / {probs[2]:.1%}"

        self.batch_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
        display_text = text[:80] + ("..." if len(text) > 80 else "")
        self.batch_table.setItem(row_idx, 1, QTableWidgetItem(display_text))
        self.batch_table.setItem(row_idx, 2, QTableWidgetItem(f"{emoji[pred]} {label}"))
        self.batch_table.setItem(row_idx, 3, QTableWidgetItem(prob_str))

        self.batch_texts.append(text)
        self.batch_results.append((pred, probs))

    def _on_batch_error(self, msg):
        QMessageBox.warning(self, "分析出错", f"处理文件时发生错误:\n{msg}")
        self.batch_run_btn.setEnabled(True)
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(1)

    def _on_batch_finished(self):
        self.batch_run_btn.setEnabled(True)
        self.export_btn.show()
        self.export_btn.setEnabled(True)

        # 统计
        total = len(self.batch_results)
        neg = sum(1 for p, _ in self.batch_results if p == 0)
        neu = sum(1 for p, _ in self.batch_results if p == 1)
        pos = sum(1 for p, _ in self.batch_results if p == 2)

        summary = (
            f"📊 分析完成！共 {total} 条评论\n"
            f"    🔴 负面: {neg} 条 ({neg/total*100:.1f}%)\n"
            f"    🟡 中性: {neu} 条 ({neu/total*100:.1f}%)\n"
            f"    🟢 正面: {pos} 条 ({pos/total*100:.1f}%)"
        )
        self.batch_summary.setText(summary)
        self.batch_summary.show()
        self.statusbar.showMessage(f"✅ 批量分析完成 | {summary.split(chr(10))[0]}")

    def _on_export(self):
        import csv
        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "batch_result.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["序号", "评论文本", "预测结果", "负面概率", "中性概率", "正面概率"])
            for i, (text, (pred, probs)) in enumerate(zip(self.batch_texts, self.batch_results)):
                writer.writerow([
                    i + 1, text, LABEL_NAMES[pred],
                    f"{probs[0]:.4f}", f"{probs[1]:.4f}", f"{probs[2]:.4f}",
                ])
        QMessageBox.information(self, "导出成功", f"结果已保存到:\n{path}")

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
