#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型預測準確率計算方式說明和實作
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_regression_accuracy_methods(y_true, y_pred):
    """
    計算回歸問題的多種準確率指標

    Parameters:
    y_true: 實際值
    y_pred: 預測值

    Returns:
    dict: 包含各種準確率指標的字典
    """

    print("=== 回歸問題準確率計算方法 ===")

    # 1. 分類準確率 (將連續值轉為離散值)
    print("\n1. 分類準確率 (Classification Accuracy)")
    print("   將連續的品質分數視為分類問題")

    # 四捨五入到最近的整數
    y_pred_rounded = np.round(y_pred).astype(int)
    y_true_rounded = np.round(y_true).astype(int)

    # 限制在有效範圍內 (3-8)
    y_pred_clipped = np.clip(y_pred_rounded, 3, 8)
    y_true_clipped = np.clip(y_true_rounded, 3, 8)

    exact_accuracy = accuracy_score(y_true_clipped, y_pred_clipped)
    print(f"   完全正確預測: {exact_accuracy:.4f} ({exact_accuracy*100:.2f}%)")

    # 計算誤差在±1範圍內的準確率
    diff = np.abs(y_true_clipped - y_pred_clipped)
    within_one_accuracy = np.mean(diff <= 1)
    print(
        f"   誤差±1範圍內: {within_one_accuracy:.4f} ({within_one_accuracy*100:.2f}%)"
    )

    # 計算誤差在±2範圍內的準確率
    within_two_accuracy = np.mean(diff <= 2)
    print(
        f"   誤差±2範圍內: {within_two_accuracy:.4f} ({within_two_accuracy*100:.2f}%)"
    )

    # 2. R² (決定係數)
    print("\n2. R2 (決定係數 / Coefficient of Determination)")
    r2 = r2_score(y_true, y_pred)
    print(f"   R2 = {r2:.4f}")
    print(f"   解釋變異比例: {r2*100:.2f}%")
    print("   解釋: R2越接近1表示模型解釋能力越好")

    # 3. RMSE (均方根誤差)
    print("\n3. RMSE (均方根誤差 / Root Mean Square Error)")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"   RMSE = {rmse:.4f}")
    print("   解釋: RMSE越小表示預測誤差越小")

    # 4. MAE (平均絕對誤差)
    print("\n4. MAE (平均絕對誤差 / Mean Absolute Error)")
    mae = mean_absolute_error(y_true, y_pred)
    print(f"   MAE = {mae:.4f}")
    print("   解釋: MAE越小表示平均預測誤差越小")

    # 5. 相對誤差
    print("\n5. 相對誤差 (Relative Error)")
    relative_error = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    print(f"   平均相對誤差: {relative_error:.2f}%")

    # 6. 各品質等級的準確率
    print("\n6. 各品質等級預測準確率")
    quality_levels = sorted(y_true_clipped.unique())
    for level in quality_levels:
        mask = y_true_clipped == level
        if np.sum(mask) > 0:
            level_accuracy = accuracy_score(y_true_clipped[mask], y_pred_clipped[mask])
            count = np.sum(mask)
            print(
                f"   品質 {level}: {level_accuracy:.4f} ({level_accuracy*100:.2f}%) - {count} 筆"
            )

    return {
        "exact_accuracy": exact_accuracy,
        "within_one_accuracy": within_one_accuracy,
        "within_two_accuracy": within_two_accuracy,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "relative_error": relative_error,
    }


def demonstrate_accuracy_calculation():
    """示範準確率計算"""
    print("=== 模型預測準確率計算示範 ===")

    # 載入保留異常值版本的模型和資料
    try:
        # 載入模型
        model = joblib.load("results_no_outliers/models/ridge_model.pkl")

        # 載入測試資料
        X_test = pd.read_csv("processed_data_no_outliers/X_test.csv")
        y_test = pd.read_csv("processed_data_no_outliers/y_test.csv").squeeze()

        # 進行預測
        y_pred = model.predict(X_test)

        print(f"測試集樣本數: {len(y_test)}")
        print(f"實際值範圍: {y_test.min()} - {y_test.max()}")
        print(f"預測值範圍: {y_pred.min():.2f} - {y_pred.max():.2f}")

        # 計算各種準確率指標
        accuracy_metrics = calculate_regression_accuracy_methods(y_test, y_pred)

        return accuracy_metrics

    except FileNotFoundError:
        print("找不到模型檔案，使用模擬資料示範")

        # 使用模擬資料
        np.random.seed(42)
        y_true = np.random.choice(
            [3, 4, 5, 6, 7, 8], size=100, p=[0.01, 0.05, 0.4, 0.4, 0.1, 0.03]
        )
        y_pred = y_true + np.random.normal(0, 0.5, size=100)

        print(f"模擬資料樣本數: {len(y_true)}")
        print(f"實際值範圍: {y_true.min()} - {y_true.max()}")
        print(f"預測值範圍: {y_pred.min():.2f} - {y_pred.max():.2f}")

        # 計算各種準確率指標
        accuracy_metrics = calculate_regression_accuracy_methods(y_true, y_pred)

        return accuracy_metrics


def accuracy_interpretation_guide():
    """準確率解釋指南"""
    print("\n=== 準確率指標解釋指南 ===")

    guide = """
## 回歸問題準確率指標解釋

### 1. 分類準確率 (Classification Accuracy)
- **完全正確**: 預測值四捨五入後與實際值完全相同
- **誤差±1**: 預測值與實際值相差不超過1
- **誤差±2**: 預測值與實際值相差不超過2
- **適用場景**: 當品質分數為離散值時

### 2. R2 (決定係數)
- **範圍**: 0 到 1 (可能為負值)
- **解釋**: 
  - R2 = 1: 完美預測
  - R2 = 0: 模型不比平均值預測好
  - R2 < 0: 模型比平均值預測還差
- **一般標準**: R2 > 0.3 為可接受，R2 > 0.5 為良好

### 3. RMSE (均方根誤差)
- **單位**: 與目標變數相同
- **解釋**: 預測誤差的標準差
- **特點**: 對大誤差敏感
- **比較**: 不同模型的RMSE可直接比較

### 4. MAE (平均絕對誤差)
- **單位**: 與目標變數相同
- **解釋**: 平均預測誤差
- **特點**: 對異常值不敏感
- **比較**: 不同模型的MAE可直接比較

### 5. 相對誤差
- **單位**: 百分比
- **解釋**: 預測誤差相對於實際值的比例
- **適用**: 比較不同量級的預測

## 選擇準確率指標的建議

### 對於葡萄酒品質預測:
1. **主要指標**: 分類準確率 (完全正確 + 誤差±1)
2. **輔助指標**: R2, RMSE, MAE
3. **目標**: 完全正確 > 60%, 誤差±1 > 95%

### 模型比較:
- 使用多個指標綜合評估
- 關注業務相關的指標
- 考慮模型的穩定性
    """

    print(guide)


def main():
    """主執行函數"""
    print("模型預測準確率計算方式說明")
    print("=" * 50)

    # 示範準確率計算
    accuracy_metrics = demonstrate_accuracy_calculation()

    # 提供解釋指南
    accuracy_interpretation_guide()

    print("\n=== 總結 ===")
    print("回歸問題的準確率計算需要考慮多個指標:")
    print("1. 分類準確率 - 適用於離散目標")
    print("2. R2 - 解釋變異能力")
    print("3. RMSE - 預測誤差大小")
    print("4. MAE - 平均預測誤差")
    print("5. 相對誤差 - 誤差比例")

    return accuracy_metrics


if __name__ == "__main__":
    metrics = main()
