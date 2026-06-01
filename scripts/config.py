"""
全局配置文件
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, "data")
NEUTRAL_CSV = os.path.join(DATA_DIR, "ai_neutral.csv")
SENTI_CSV = os.path.join(DATA_DIR, "ChnSentiCorp_htl_all.csv")

# 预处理后数据保存路径
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 模型保存路径
# 打包后模型放在 exe 同级的 models/ 目录
def _get_models_dir():
    import sys
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), 'models')
    else:
        return os.path.join(BASE_DIR, 'models')

MODEL_DIR = _get_models_dir()
STAGE1_MODEL_DIR = os.path.join(MODEL_DIR, "stage1_neutral")
STAGE2_MODEL_DIR = os.path.join(MODEL_DIR, "stage2_sentiment")
os.makedirs(STAGE1_MODEL_DIR, exist_ok=True)
os.makedirs(STAGE2_MODEL_DIR, exist_ok=True)

# 日志 & 结果目录
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ========== BERT 模型配置 ==========
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 128
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64

# ========== 训练超参数 ==========
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

# ========== 数据集划分 ==========
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# ========== 标签映射 ==========
# 三分类: 0=负面, 1=中性, 2=正面
LABEL_NAMES = {0: "负面", 1: "中性", 2: "正面"}

# GPU
DEVICE = "cuda"  # 若无 GPU 则改为 "cpu"
