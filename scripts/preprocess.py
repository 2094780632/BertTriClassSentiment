"""
数据预处理脚本：
1. 合并两个 CSV，统一标签：0=负面, 1=中性, 2=正面
2. 清洗文本（去除空值、去重）
3. 按 8:1:1 划分 train / val / test（分层抽样）
4. 为 Stage1 生成二分类标签（中性 vs 非中性）
5. 为 Stage2 生成二分类标签（负面 vs 正面，仅非中性样本）
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import (
    NEUTRAL_CSV, SENTI_CSV, PROCESSED_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)
import re


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白、统一换行"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text


def load_and_merge() -> pd.DataFrame:
    """加载两个数据集并合并为统一格式"""
    # 1. 中性数据：ai_neutral.csv → label 全部为 1（中性）
    df_neutral = pd.read_csv(NEUTRAL_CSV)
    df_neutral = df_neutral.rename(columns={"text": "review"})
    df_neutral["label"] = 1  # 中性
    print(f"[中性数据] {len(df_neutral)} 条")

    # 2. 情感数据：ChnSentiCorp_htl_all.csv
    #    label=0 → 负面(0), label=1 → 正面(2)
    df_senti = pd.read_csv(SENTI_CSV)
    df_senti["label"] = df_senti["label"].map({0: 0, 1: 2})  # 0=负面, 2=正面
    print(f"[情感数据] {len(df_senti)} 条 (负面: {(df_senti['label']==0).sum()}, 正面: {(df_senti['label']==2).sum()})")

    # 3. 合并
    df = pd.concat([df_neutral, df_senti], ignore_index=True)
    df = df[["review", "label"]]  # 只保留需要的列

    print(f"[合并后] 总计 {len(df)} 条")
    print(f"  负面(0): {(df['label']==0).sum()}")
    print(f"  中性(1): {(df['label']==1).sum()}")
    print(f"  正面(2): {(df['label']==2).sum()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗：去除空文本、去重"""
    df["review"] = df["review"].apply(clean_text)
    before = len(df)
    df = df[df["review"] != ""].copy()
    df = df.drop_duplicates(subset=["review"]).copy()
    after = len(df)
    print(f"[清洗] 去除空文本和重复后：{before} → {after}")
    return df


def split_dataset(df: pd.DataFrame):
    """8:1:1 分层划分 train/val/test"""
    # 先拆出 train (80%)，剩余 20%
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["label"], random_state=RANDOM_SEED,
    )
    # 再从剩余中拆 val 和 test 各一半
    val_df, test_df = train_test_split(
        temp_df, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        stratify=temp_df["label"], random_state=RANDOM_SEED,
    )
    print(f"[划分] train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def build_stage1_labels(df: pd.DataFrame):
    """
    Stage1 二分类：中性(1) vs 非中性(0)
    原标签 1 → 1（中性），原标签 0,2 → 0（非中性）
    """
    df = df.copy()
    df["stage1_label"] = (df["label"] == 1).astype(int)
    return df


def build_stage2_data(df: pd.DataFrame):
    """
    Stage2 二分类：仅保留非中性样本，负面(0) vs 正面(1)
    原标签 0 → 0（负面），原标签 2 → 1（正面）
    """
    df = df[df["label"] != 1].copy()  # 去掉中性
    df["stage2_label"] = (df["label"] == 2).astype(int)  # 正面=1, 负面=0
    return df


def main():
    print("=" * 60)
    print("数据预处理开始")
    print("=" * 60)

    # 加载合并
    df = load_and_merge()

    # 清洗
    df = clean_data(df)

    # 划分
    train_df, val_df, test_df = split_dataset(df)

    # --- 保存三分类原始数据 ---
    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val_df.to_csv(f"{PROCESSED_DIR}/val.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)
    print("[保存] train.csv / val.csv / test.csv 已保存")

    # --- Stage1: 中性 vs 非中性 ---
    for name, subset in [("train", train_df), ("val", val_df), ("test", test_df)]:
        s1 = build_stage1_labels(subset)
        s1[["review", "stage1_label"]].to_csv(
            f"{PROCESSED_DIR}/stage1_{name}.csv", index=False
        )
    print("[保存] stage1_train/val/test.csv 已保存")

    # --- Stage2: 负面 vs 正面（仅非中性） ---
    for name, subset in [("train", train_df), ("val", val_df), ("test", test_df)]:
        s2 = build_stage2_data(subset)
        s2[["review", "stage2_label"]].to_csv(
            f"{PROCESSED_DIR}/stage2_{name}.csv", index=False
        )
    print("[保存] stage2_train/val/test.csv 已保存")

    print("=" * 60)
    print("数据预处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
