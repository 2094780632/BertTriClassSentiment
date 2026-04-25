"""
级联集成模型评估脚本
====================
评估三分类情感分析模型的性能

使用方法：
    python scripts/evaluate_cascade_model.py

数据集：
    - ChnSentiCorp_htl_all.csv (酒店评论，正负二分类)
    - waimai_10k.csv (外卖评论，正负二分类)
    - neutral_train_strict.csv (中性评论)

输出：
    output/cascade_evaluation/
    ├── evaluation_report.txt
    └── figures/
        ├── 01_confusion_matrix.png
        ├── 02_metrics_bar.png
        ├── 03_confidence_distribution.png
        ├── 04_error_analysis.png
        ├── 05_dashboard.png
        ├── 06_threshold_analysis.png
        └── 07_source_performance.png
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from pathlib import Path
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 导入预测器
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from predict import CascadePredictor, Config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# ============ 路径配置 ============
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output' / 'cascade_evaluation'
FIG_DIR = OUTPUT_DIR / 'figures'

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 标签映射
LABELS_MAP = {0: '负面', 1: '正面', 2: '中性'}
LABELS_MAP_REVERSE = {'负面': 0, '正面': 1, '中性': 2}
COLORS = {'负面': '#e74c3c', '正面': '#2ecc71', '中性': '#3498db'}

print("="*60)
print("级联集成模型评估系统")
print("="*60)

# ==================== 1. 加载测试数据 ====================
print("\n[1/5] 加载测试数据...")

def load_test_data():
    """加载并构建三分类测试集"""

    # 加载正面样本
    htl_df = pd.read_csv(DATA_DIR / 'ChnSentiCorp_htl_all.csv')
    waimai_df = pd.read_csv(DATA_DIR / 'waimai_10k.csv')

    # 合并正面和负面样本
    pos_neg_df = pd.concat([htl_df, waimai_df], ignore_index=True)
    pos_neg_df['category'] = pos_neg_df['label'].map({0: '负面', 1: '正面'})

    # 加载中性样本
    neutral_df = pd.read_csv(DATA_DIR / 'neutral_train_strict.csv')
    neutral_df['category'] = '中性'
    neutral_df['label'] = 2  # 中性标签为2

    # 合并所有数据
    all_df = pd.concat([
        pos_neg_df[['review', 'label', 'category']],
        neutral_df[['review', 'label', 'category']]
    ], ignore_index=True)

    # 采样以保证平衡（可选）
    print(f"  正面样本: {sum(all_df['label'] == 1)}")
    print(f"  负面样本: {sum(all_df['label'] == 0)}")
    print(f"  中性样本: {sum(all_df['label'] == 2)}")

    # 随机打乱
    all_df = all_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 划分测试集（使用20%作为测试集）
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(
        all_df, test_size=0.2, random_state=42,
        stratify=all_df['label']
    )

    print(f"\n测试集大小: {len(test_df)} 条")
    for label, name in LABELS_MAP.items():
        count = sum(test_df['label'] == label)
        print(f"  {name}: {count} 条 ({count/len(test_df)*100:.1f}%)")

    return test_df.reset_index(drop=True)

test_df = load_test_data()

# ==================== 2. 模型预测 ====================
print("\n[2/5] 加载模型并进行预测...")

predictor = CascadePredictor()

# 批量预测
y_true = []
y_pred = []
confidences = []
error_details = []

print("正在进行预测...")
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    text = row['review']
    true_label = row['label']

    result = predictor.predict(text)

    y_true.append(true_label)
    y_pred.append(result['label'])
    confidences.append(result['confidence'])

    if true_label != result['label']:
        error_details.append({
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'true_label': LABELS_MAP[true_label],
            'pred_label': result['label_name'],
            'confidence': result['confidence'],
            'details': str(result['details'])
        })

y_true = np.array(y_true)
y_pred = np.array(y_pred)
confidences = np.array(confidences)

print(f"\n✓ 预测完成")
print(f"  正确预测: {sum(y_true == y_pred)}/{len(y_true)}")
print(f"  错误预测: {sum(y_true != y_pred)}/{len(y_true)}")

# ==================== 3. 计算指标 ====================
print("\n[3/5] 计算评估指标...")

def calculate_metrics(y_true, y_pred):
    """计算各项评估指标"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # 每个类别的指标
    for label, name in LABELS_MAP.items():
        metrics[f'{name}_precision'] = precision_score(
            y_true, y_pred, labels=[label], average=None, zero_division=0
        )[0]
        metrics[f'{name}_recall'] = recall_score(
            y_true, y_pred, labels=[label], average=None, zero_division=0
        )[0]
        metrics[f'{name}_f1'] = f1_score(
            y_true, y_pred, labels=[label], average=None, zero_division=0
        )[0]

    return metrics

metrics = calculate_metrics(y_true, y_pred)

# ==================== 4. 可视化类 ====================
class CascadeEvalVisualizer:
    def __init__(self, y_true, y_pred, confidences, test_df):
        self.y_true = y_true
        self.y_pred = y_pred
        self.confidences = confidences
        self.test_df = test_df
        self.correct = (y_true == y_pred)

    def plot_confusion_matrix(self, save_path=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1, 2])

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[LABELS_MAP[0], LABELS_MAP[1], LABELS_MAP[2]],
                    yticklabels=[LABELS_MAP[0], LABELS_MAP[1], LABELS_MAP[2]],
                    ax=ax, annot_kws={'size': 14})

        ax.set_xlabel('预测标签', fontsize=13)
        ax.set_ylabel('真实标签', fontsize=13)
        ax.set_title('混淆矩阵 - 级联模型预测结果', fontsize=15, fontweight='bold')

        # 添加百分比
        total = cm.sum()
        for i in range(3):
            for j in range(3):
                pct = cm[i, j] / total * 100
                ax.text(j+0.5, i+0.7, f'({pct:.1f}%)',
                       ha='center', va='center', fontsize=10, color='gray')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  保存: {save_path}")
        plt.show()
        return fig

    def plot_metrics_bar(self, save_path=None):
        """绘制各类别指标对比"""
        categories = ['负面', '正面', '中性']
        precision = [metrics[f'{c}_precision'] for c in categories]
        recall = [metrics[f'{c}_recall'] for c in categories]
        f1 = [metrics[f'{c}_f1'] for c in categories]

        x = np.arange(len(categories))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width, precision, width, label='精确率', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='召回率', color='#2ecc71')
        bars3 = ax.bar(x + width, f1, width, label='F1分数', color='#e74c3c')

        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_title('各类别评估指标对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.1)

        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  保存: {save_path}")
        plt.show()
        return fig

    def plot_confidence_distribution(self, save_path=None):
        """绘制置信度分布"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for idx, (label, name) in enumerate(LABELS_MAP.items()):
            mask = self.y_true == label
            correct_mask = mask & self.correct
            wrong_mask = mask & ~self.correct

            ax = axes[idx]

            if correct_mask.sum() > 0:
                ax.hist(self.confidences[correct_mask], bins=20, alpha=0.7,
                       label=f'正确 ({correct_mask.sum()})', color='#2ecc71', edgecolor='white')

            if wrong_mask.sum() > 0:
                ax.hist(self.confidences[wrong_mask], bins=20, alpha=0.7,
                       label=f'错误 ({wrong_mask.sum()})', color='#e74c3c', edgecolor='white')

            ax.set_xlabel('置信度', fontsize=11)
            ax.set_ylabel('样本数', fontsize=11)
            ax.set_title(f'{name}样本置信度分布', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.set_xlim(0, 1)

        plt.suptitle('各类别预测置信度分布', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  保存: {save_path}")
        plt.show()
        return fig

    def plot_error_analysis(self, save_path=None):
        """绘制错误分析图"""
        # 修复：重新创建 error_df 并重置索引
        error_indices = np.where(self.y_true != self.y_pred)[0]
        error_df = self.test_df.iloc[error_indices].copy()

        if len(error_df) == 0:
            print("没有错误样本，跳过错误分析")
            return None

        # 获取对应的预测标签
        error_true_labels = self.y_true[error_indices]
        error_pred_labels = self.y_pred[error_indices]

        # 创建错误类型
        error_types = []
        for i in range(len(error_df)):
            true_name = LABELS_MAP[error_true_labels[i]]
            pred_name = LABELS_MAP[error_pred_labels[i]]
            error_types.append(f"{true_name}→{pred_name}")

        error_df['error_type'] = error_types
        error_counts = error_df['error_type'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(error_counts)))
        bars = ax.barh(range(len(error_counts)), error_counts.values, color=colors)

        ax.set_yticks(range(len(error_counts)))
        ax.set_yticklabels(error_counts.index)
        ax.set_xlabel('错误样本数', fontsize=12)
        ax.set_title('错误类型分析（真实→预测）', fontsize=14, fontweight='bold')

        for i, (bar, val) in enumerate(zip(bars, error_counts.values)):
            ax.text(val + 0.5, i, str(val), va='center', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  保存: {save_path}")
        plt.show()
        return fig

    def plot_dashboard(self, metrics, save_path=None):
        """绘制整体指标仪表盘"""
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 准确率
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, f'{metrics["accuracy"]:.2%}',
                ha='center', va='center', fontsize=40, fontweight='bold', color='#2c3e50')
        ax1.text(0.5, 0.2, '整体准确率', ha='center', va='center', fontsize=14)
        ax1.axis('off')

        # Macro F1
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, f'{metrics["macro_f1"]:.3f}',
                ha='center', va='center', fontsize=40, fontweight='bold', color='#e74c3c')
        ax2.text(0.5, 0.2, 'Macro F1', ha='center', va='center', fontsize=14)
        ax2.axis('off')

        # 样本数
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.5, f'{len(self.y_true)}',
                ha='center', va='center', fontsize=40, fontweight='bold', color='#3498db')
        ax3.text(0.5, 0.2, '评估样本数', ha='center', va='center', fontsize=14)
        ax3.axis('off')

        # 各类别F1
        ax4 = fig.add_subplot(gs[1:, :])
        categories = ['负面', '正面', '中性']
        f1_scores = [metrics[f'{c}_f1'] for c in categories]
        colors_list = [COLORS[c] for c in categories]

        bars = ax4.bar(categories, f1_scores, color=colors_list, alpha=0.8, edgecolor='black')
        ax4.set_ylabel('F1 分数', fontsize=12)
        ax4.set_title('各类别 F1 分数', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1)

        for bar, val in zip(bars, f1_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.suptitle('级联模型评估仪表盘', fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  保存: {save_path}")
        plt.show()
        return fig

    def plot_threshold_analysis(self, save_path=None):
        """阈值分析：不同中性阈值下的性能变化"""
        thresholds = np.arange(0.3, 0.9, 0.05)
        accuracies = []
        macro_f1s = []

        print("\n  分析不同中性阈值对性能的影响...")

        # 预处理所有文本和真实标签
        texts = self.test_df['review'].tolist()
        y_true = self.y_true

        for thresh in thresholds:
            print(f"    测试阈值: {thresh}")
            y_pred_temp = []

            # 直接调用预测，但避免重复加载模型
            for text in texts:
                # 这里需要修改 predict 方法，让它能接受临时阈值参数
                # 临时解决方案：直接修改 Config 的阈值
                original_threshold = Config.NEUTRAL_THRESHOLD
                Config.NEUTRAL_THRESHOLD = thresh

                result = predictor.predict(text)
                y_pred_temp.append(result['label'])

                # 恢复原阈值
                Config.NEUTRAL_THRESHOLD = original_threshold

            acc = accuracy_score(y_true, y_pred_temp)
            f1 = f1_score(y_true, y_pred_temp, average='macro')
            accuracies.append(acc)
            macro_f1s.append(f1)
            print(f"      准确率: {acc:.4f}, F1: {f1:.4f}")

        # 绘图代码保持不变...
    def generate_all_plots(self, metrics):
        """生成所有图表"""
        print("\n[4/5] 生成可视化图表...")

        self.plot_confusion_matrix(FIG_DIR / '01_confusion_matrix.png')
        self.plot_metrics_bar(FIG_DIR / '02_metrics_bar.png')
        self.plot_confidence_distribution(FIG_DIR / '03_confidence_distribution.png')
        self.plot_error_analysis(FIG_DIR / '04_error_analysis.png')
        self.plot_dashboard(metrics, FIG_DIR / '05_dashboard.png')
        self.plot_threshold_analysis(FIG_DIR / '06_threshold_analysis.png')

        print(f"\n✓ 所有图表已保存到: {FIG_DIR}")

# ==================== 5. 生成报告 ====================
def generate_report(metrics, y_true, y_pred, test_df, error_count):
    """生成文本评估报告"""
    report = []
    report.append("="*70)
    report.append("级联集成模型评估报告")
    report.append("="*70)
    report.append(f"\n评估时间: {pd.Timestamp.now()}")
    report.append(f"测试集大小: {len(y_true)} 条")
    report.append(f"正确预测: {sum(y_true == y_pred)} 条")
    report.append(f"错误预测: {sum(y_true != y_pred)} 条")
    report.append(f"整体准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

    report.append(f"\n宏观平均 (Macro):")
    report.append(f"  精确率: {metrics['macro_precision']:.4f}")
    report.append(f"  召回率: {metrics['macro_recall']:.4f}")
    report.append(f"  F1分数: {metrics['macro_f1']:.4f}")

    report.append(f"\n加权平均 (Weighted):")
    report.append(f"  精确率: {metrics['weighted_precision']:.4f}")
    report.append(f"  召回率: {metrics['weighted_recall']:.4f}")
    report.append(f"  F1分数: {metrics['weighted_f1']:.4f}")

    report.append(f"\n各类别详细指标:")
    for name in ['负面', '正面', '中性']:
        report.append(f"\n{name}:")
        report.append(f"  精确率: {metrics[f'{name}_precision']:.4f}")
        report.append(f"  召回率: {metrics[f'{name}_recall']:.4f}")
        report.append(f"  F1分数: {metrics[f'{name}_f1']:.4f}")
        report.append(f"  样本数: {sum(y_true == LABELS_MAP_REVERSE[name])}")

    report.append(f"\n分类报告:")
    report.append(classification_report(y_true, y_pred,
                                       target_names=['负面', '正面', '中性'],
                                       digits=4))

    report.append(f"\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report.append("           预测负面  预测正面  预测中性")
    report.append(f"实际负面:   {cm[0,0]:6d}   {cm[0,1]:6d}   {cm[0,2]:6d}")
    report.append(f"实际正面:   {cm[1,0]:6d}   {cm[1,1]:6d}   {cm[1,2]:6d}")
    report.append(f"实际中性:   {cm[2,0]:6d}   {cm[2,1]:6d}   {cm[2,2]:6d}")

    report.append(f"\n模型配置:")
    report.append(f"  中性模型: {Config.NEUTRAL_MODEL_PATH}")
    report.append(f"  情感模型: {Config.SENTIMENT_MODEL_DIR}")
    report.append(f"  中性阈值: {Config.NEUTRAL_THRESHOLD}")
    report.append(f"  最大长度: {Config.MAX_LEN}")

    # 保存报告
    report_text = "\n".join(report)
    report_file = OUTPUT_DIR / 'evaluation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✓ 报告已保存: {report_file}")

    return report_text

# ==================== 6. 主程序 ====================
if __name__ == '__main__':
    # 初始化可视化器
    visualizer = CascadeEvalVisualizer(y_true, y_pred, confidences, test_df)

    # 生成报告
    generate_report(metrics, y_true, y_pred, test_df, len(error_details))

    # 生成可视化
    visualizer.generate_all_plots(metrics)

    # 保存错误样本
    if error_details:
        error_df = pd.DataFrame(error_details)
        error_df.to_csv(OUTPUT_DIR / 'error_samples.csv', index=False, encoding='utf-8-sig')
        print(f"\n✓ 错误样本已保存: {OUTPUT_DIR / 'error_samples.csv'}")

    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)
    print(f"输出目录: {OUTPUT_DIR}")