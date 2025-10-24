#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析最佳模型的預測準確率
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def analyze_prediction_accuracy():
    """分析預測準確率"""
    print("=== 分析最佳模型預測準確率 ===")

    # 載入資料
    X_test = pd.read_csv("processed_data/X_test.csv")
    y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

    # 載入最佳模型 (Ridge)
    best_model = joblib.load("results/models/ridge_model.pkl")

    # 預測
    y_pred = best_model.predict(X_test)

    # 將連續值轉換為整數 (四捨五入)
    y_pred_rounded = np.round(y_pred).astype(int)

    # 確保預測值在合理範圍內 (3-8)
    y_pred_rounded = np.clip(y_pred_rounded, 3, 8)

    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred_rounded)

    print(f"測試集樣本數: {len(y_test)}")
    print(f"預測準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 詳細分析
    print(f"\n實際值分布:")
    print(y_test.value_counts().sort_index())

    print(f"\n預測值分布:")
    print(pd.Series(y_pred_rounded).value_counts().sort_index())

    # 分類報告
    print(f"\n分類報告:")
    print(classification_report(y_test, y_pred_rounded))

    # 混淆矩陣
    print(f"\n混淆矩陣:")
    cm = confusion_matrix(y_test, y_pred_rounded)
    print(cm)

    # 分析預測誤差
    errors = np.abs(y_test - y_pred_rounded)
    print(f"\n預測誤差分析:")
    print(f"平均絕對誤差: {errors.mean():.4f}")
    print(f"誤差標準差: {errors.std():.4f}")
    print(f"完全正確預測: {(errors == 0).sum()} 筆 ({(errors == 0).mean()*100:.2f}%)")
    print(f"誤差 ±1: {(errors <= 1).sum()} 筆 ({(errors <= 1).mean()*100:.2f}%)")
    print(f"誤差 ±2: {(errors <= 2).sum()} 筆 ({(errors <= 2).mean()*100:.2f}%)")

    # 分析不同品質等級的準確率
    print(f"\n各品質等級預測準確率:")
    for quality in sorted(y_test.unique()):
        mask = y_test == quality
        if mask.sum() > 0:
            level_accuracy = accuracy_score(y_test[mask], y_pred_rounded[mask])
            print(
                f"品質 {quality}: {level_accuracy:.4f} ({level_accuracy*100:.2f}%) - {mask.sum()} 筆"
            )

    return accuracy, y_test, y_pred_rounded


def main():
    """主執行函數"""
    accuracy, y_test, y_pred_rounded = analyze_prediction_accuracy()

    print(f"\n=== 結論 ===")
    if accuracy >= 0.8:
        print(f"[成功] 預測準確率達到 {accuracy*100:.2f}%，超過 80% 的目標！")
    else:
        print(f"[未達標] 預測準確率為 {accuracy*100:.2f}%，未達到 80% 的目標。")
        print("建議進行以下改進:")
        print("1. 特徵工程優化")
        print("2. 模型超參數調優")
        print("3. 嘗試其他演算法 (如隨機森林、XGBoost)")
        print("4. 集成學習方法")


if __name__ == "__main__":
    main()
