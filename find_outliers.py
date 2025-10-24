#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出在資料清理階段被移除的115筆異常值
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def find_outliers():
    """找出被移除的異常值"""
    print("=== 找出被移除的異常值 ===")

    # 載入原始資料
    df_original = pd.read_csv("datasets/WineQT.csv")
    print(f"原始資料: {df_original.shape[0]} 筆")

    # 載入處理後資料
    df_processed = pd.read_csv("processed_data/wine_processed.csv")
    print(f"處理後資料: {df_processed.shape[0]} 筆")

    # 計算被移除的資料數量
    removed_count = df_original.shape[0] - df_processed.shape[0]
    print(f"被移除的資料: {removed_count} 筆")

    # 使用相同的Isolation Forest參數重新識別異常值
    print("\n=== 重新識別異常值 ===")

    # 準備數值特徵 (排除ID欄位)
    numeric_features = df_original.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != "Id"]

    print(f"使用的特徵: {numeric_features}")

    # 使用相同的參數
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = iso_forest.fit_predict(df_original[numeric_features])

    # 找出異常值的索引
    outlier_indices = np.where(outlier_labels == -1)[0]
    print(f"識別出的異常值數量: {len(outlier_indices)}")

    # 顯示異常值的詳細資訊
    print(f"\n=== 異常值詳細資訊 ===")
    outliers_df = df_original.iloc[outlier_indices].copy()
    outliers_df["原始索引"] = outliers_df.index
    outliers_df["異常值標記"] = "異常值"

    # 顯示前20筆異常值
    print("前20筆異常值:")
    print(outliers_df[["原始索引", "quality"] + numeric_features[:5]].head(20))

    # 顯示異常值的統計資訊
    print(f"\n=== 異常值統計 ===")
    print(f"異常值索引範圍: {outlier_indices.min()} - {outlier_indices.max()}")
    print(f"異常值品質分布:")
    print(outliers_df["quality"].value_counts().sort_index())

    # 分析異常值的特徵
    print(f"\n=== 異常值特徵分析 ===")
    for col in numeric_features:
        if col in outliers_df.columns:
            outlier_mean = outliers_df[col].mean()
            normal_mean = df_original[col].mean()
            print(f"{col}: 異常值平均={outlier_mean:.3f}, 正常值平均={normal_mean:.3f}")

    # 儲存異常值資訊
    outliers_df.to_csv("processed_data/outliers_removed.csv", index=False)
    print(f"\n異常值已儲存至 processed_data/outliers_removed.csv")

    # 顯示異常值的索引列表
    print(f"\n=== 被移除的115筆異常值索引 ===")
    print("異常值索引 (前50筆):")
    print(outlier_indices[:50])
    if len(outlier_indices) > 50:
        print("...")
        print("異常值索引 (後50筆):")
        print(outlier_indices[-50:])

    print(f"\n完整異常值索引列表:")
    print(f"outlier_indices = {outlier_indices.tolist()}")

    return outlier_indices, outliers_df


def analyze_outlier_characteristics(outliers_df, df_original):
    """分析異常值的特徵"""
    print(f"\n=== 異常值特徵分析 ===")

    # 比較異常值和正常值的統計資訊
    numeric_features = df_original.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != "Id"]

    comparison_data = []

    for col in numeric_features:
        if col in outliers_df.columns:
            outlier_mean = outliers_df[col].mean()
            outlier_std = outliers_df[col].std()
            normal_mean = df_original[col].mean()
            normal_std = df_original[col].std()

            comparison_data.append(
                {
                    "特徵": col,
                    "異常值平均": outlier_mean,
                    "異常值標準差": outlier_std,
                    "正常值平均": normal_mean,
                    "正常值標準差": normal_std,
                    "差異": outlier_mean - normal_mean,
                }
            )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("差異", key=abs, ascending=False)

    print("異常值與正常值比較 (按差異大小排序):")
    print(comparison_df.to_string(index=False, float_format="%.3f"))

    return comparison_df


if __name__ == "__main__":
    outlier_indices, outliers_df = find_outliers()

    # 載入原始資料進行比較分析
    df_original = pd.read_csv("datasets/WineQT.csv")
    comparison_df = analyze_outlier_characteristics(outliers_df, df_original)
