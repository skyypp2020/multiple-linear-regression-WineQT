#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用驗證資料集進行各模型的 Accuracy 驗證
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class ValidationAccuracyAnalyzer:
    """驗證資料集 Accuracy 分析器"""

    def __init__(self):
        """初始化分析器"""
        self.models = {}
        self.results = {}

    def load_data(self):
        """載入資料"""
        print("=== 載入驗證資料 ===")

        # 載入驗證資料
        self.X_val = pd.read_csv("processed_data/X_val.csv")
        self.y_val = pd.read_csv("processed_data/y_val.csv").squeeze()

        # 載入測試資料 (用於比較)
        self.X_test = pd.read_csv("processed_data/X_test.csv")
        self.y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

        print(f"驗證集: {self.X_val.shape[0]} 筆 x {self.X_val.shape[1]} 特徵")
        print(f"測試集: {self.X_test.shape[0]} 筆 x {self.X_test.shape[1]} 特徵")

        return self.X_val, self.y_val, self.X_test, self.y_test

    def load_models(self):
        """載入所有模型"""
        print("\n=== 載入所有模型 ===")

        # Baseline 模型
        self.models["Baseline_LinearRegression"] = joblib.load(
            "results/models/linearregression_model.pkl"
        )
        self.models["Baseline_Ridge"] = joblib.load("results/models/ridge_model.pkl")
        self.models["Baseline_Lasso"] = joblib.load("results/models/lasso_model.pkl")
        self.models["Baseline_ElasticNet"] = joblib.load(
            "results/models/elasticnet_model.pkl"
        )

        # 特徵選擇後的模型
        self.models["Selected_LinearRegression"] = joblib.load(
            "results/models/feature_selected/linearregression_selected_model.pkl"
        )
        self.models["Selected_Ridge"] = joblib.load(
            "results/models/feature_selected/ridge_selected_model.pkl"
        )
        self.models["Selected_Lasso"] = joblib.load(
            "results/models/feature_selected/lasso_selected_model.pkl"
        )
        self.models["Selected_ElasticNet"] = joblib.load(
            "results/models/feature_selected/elasticnet_selected_model.pkl"
        )

        print(f"已載入 {len(self.models)} 個模型:")
        for name in self.models.keys():
            print(f"- {name}")

        return self.models

    def evaluate_validation_accuracy(self):
        """評估驗證集準確率"""
        print("\n=== 驗證集準確率評估 ===")

        for name, model in self.models.items():
            print(f"\n{name} 模型:")

            # 預測
            if "Selected" in name:
                # 特徵選擇後的模型需要篩選特徵
                selected_features = pd.read_csv("results/data/selected_features.csv")[
                    "Feature"
                ].tolist()
                X_val_selected = self.X_val[selected_features]
                X_test_selected = self.X_test[selected_features]

                y_val_pred = model.predict(X_val_selected)
                y_test_pred = model.predict(X_test_selected)
            else:
                # Baseline 模型使用全部特徵
                y_val_pred = model.predict(self.X_val)
                y_test_pred = model.predict(self.X_test)

            # 四捨五入並限制範圍
            y_val_pred_rounded = np.round(y_val_pred).astype(int)
            y_val_pred_rounded = np.clip(y_val_pred_rounded, 3, 8)

            y_test_pred_rounded = np.round(y_test_pred).astype(int)
            y_test_pred_rounded = np.clip(y_test_pred_rounded, 3, 8)

            # 計算準確率
            val_accuracy = accuracy_score(self.y_val, y_val_pred_rounded)
            test_accuracy = accuracy_score(self.y_test, y_test_pred_rounded)

            # 儲存結果
            self.results[name] = {
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "val_predictions": y_val_pred_rounded,
                "test_predictions": y_test_pred_rounded,
            }

            print(f"  驗證集準確率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"  測試集準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        return self.results

    def compare_models(self):
        """比較模型性能"""
        print("\n=== 模型性能比較 ===")

        # 創建比較表格
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append(
                {
                    "Model": name,
                    "Validation_Accuracy": results["val_accuracy"],
                    "Test_Accuracy": results["test_accuracy"],
                    "Accuracy_Diff": results["test_accuracy"] - results["val_accuracy"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(
            "Validation_Accuracy", ascending=False
        )

        print("驗證集準確率排名:")
        print("=" * 80)
        print(
            f"{'模型':<25} {'驗證集準確率':<12} {'測試集準確率':<12} {'準確率差異':<10}"
        )
        print("-" * 80)

        for _, row in comparison_df.iterrows():
            print(
                f"{row['Model']:<25} {row['Validation_Accuracy']:<12.4f} {row['Test_Accuracy']:<12.4f} {row['Accuracy_Diff']:<10.4f}"
            )

        # 找出最佳模型
        best_model = comparison_df.iloc[0]
        print(f"\n最佳模型: {best_model['Model']}")
        print(
            f"驗證集準確率: {best_model['Validation_Accuracy']:.4f} ({best_model['Validation_Accuracy']*100:.2f}%)"
        )
        print(
            f"測試集準確率: {best_model['Test_Accuracy']:.4f} ({best_model['Test_Accuracy']*100:.2f}%)"
        )

        return comparison_df

    def analyze_validation_performance(self):
        """分析驗證集性能"""
        print("\n=== 驗證集性能分析 ===")

        # 分析 Baseline vs 特徵選擇後的模型
        baseline_models = [name for name in self.results.keys() if "Baseline" in name]
        selected_models = [name for name in self.results.keys() if "Selected" in name]

        print("Baseline 模型驗證集準確率:")
        for name in baseline_models:
            accuracy = self.results[name]["val_accuracy"]
            print(f"  {name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("\n特徵選擇後模型驗證集準確率:")
        for name in selected_models:
            accuracy = self.results[name]["val_accuracy"]
            print(f"  {name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # 比較 Baseline 和特徵選擇後的模型
        print("\nBaseline vs 特徵選擇後模型比較:")
        for baseline_name in baseline_models:
            model_type = baseline_name.split("_")[1]  # 取得模型類型
            selected_name = f"Selected_{model_type}"

            if selected_name in self.results:
                baseline_acc = self.results[baseline_name]["val_accuracy"]
                selected_acc = self.results[selected_name]["val_accuracy"]
                improvement = selected_acc - baseline_acc

                print(f"  {model_type}:")
                print(f"    Baseline: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
                print(f"    特徵選擇後: {selected_acc:.4f} ({selected_acc*100:.2f}%)")
                print(f"    改進: {improvement:.4f} ({improvement*100:.2f}%)")

    def visualize_results(self, comparison_df):
        """視覺化結果"""
        print("\n=== 生成驗證集準確率視覺化圖表 ===")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 驗證集準確率比較
        models = comparison_df["Model"]
        val_acc = comparison_df["Validation_Accuracy"]
        test_acc = comparison_df["Test_Accuracy"]

        x = np.arange(len(models))
        width = 0.35

        axes[0, 0].bar(
            x - width / 2, val_acc, width, label="驗證集", alpha=0.8, color="skyblue"
        )
        axes[0, 0].bar(
            x + width / 2,
            test_acc,
            width,
            label="測試集",
            alpha=0.8,
            color="lightcoral",
        )
        axes[0, 0].set_xlabel("模型")
        axes[0, 0].set_ylabel("準確率")
        axes[0, 0].set_title("驗證集 vs 測試集準確率比較")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha="right")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Baseline vs 特徵選擇後模型比較
        baseline_mask = comparison_df["Model"].str.contains("Baseline")
        selected_mask = comparison_df["Model"].str.contains("Selected")

        baseline_data = comparison_df[baseline_mask]
        selected_data = comparison_df[selected_mask]

        # 提取模型類型
        baseline_types = baseline_data["Model"].str.split("_").str[1]
        selected_types = selected_data["Model"].str.split("_").str[1]

        # 按模型類型排序
        model_types = ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]
        baseline_acc = [
            (
                baseline_data[baseline_data["Model"].str.contains(t)][
                    "Validation_Accuracy"
                ].iloc[0]
                if t in baseline_data["Model"].str.split("_").str[1].values
                else 0
            )
            for t in model_types
        ]
        selected_acc = [
            (
                selected_data[selected_data["Model"].str.contains(t)][
                    "Validation_Accuracy"
                ].iloc[0]
                if t in selected_data["Model"].str.split("_").str[1].values
                else 0
            )
            for t in model_types
        ]

        x = np.arange(len(model_types))
        axes[0, 1].bar(
            x - width / 2,
            baseline_acc,
            width,
            label="Baseline",
            alpha=0.8,
            color="lightblue",
        )
        axes[0, 1].bar(
            x + width / 2,
            selected_acc,
            width,
            label="特徵選擇後",
            alpha=0.8,
            color="lightgreen",
        )
        axes[0, 1].set_xlabel("模型類型")
        axes[0, 1].set_ylabel("驗證集準確率")
        axes[0, 1].set_title("Baseline vs 特徵選擇後模型比較")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_types)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 準確率差異分析
        accuracy_diff = comparison_df["Accuracy_Diff"]
        colors = ["green" if x > 0 else "red" for x in accuracy_diff]
        axes[1, 0].bar(models, accuracy_diff, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel("模型")
        axes[1, 0].set_ylabel("準確率差異 (測試集 - 驗證集)")
        axes[1, 0].set_title("驗證集與測試集準確率差異")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 模型性能分布
        axes[1, 1].hist(val_acc, bins=10, alpha=0.7, color="skyblue", label="驗證集")
        axes[1, 1].hist(
            test_acc, bins=10, alpha=0.7, color="lightcoral", label="測試集"
        )
        axes[1, 1].set_xlabel("準確率")
        axes[1, 1].set_ylabel("頻率")
        axes[1, 1].set_title("準確率分布")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "results/plots/validation_accuracy_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            "驗證集準確率分析圖表已儲存至 results/plots/validation_accuracy_analysis.png"
        )

    def generate_validation_report(self, comparison_df):
        """生成驗證報告"""
        print("\n=== 生成驗證報告 ===")

        # 找出最佳模型
        best_model = comparison_df.iloc[0]

        report = f"""
# WineQT 驗證集準確率分析報告

## 概述
- 分析時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 驗證集樣本數: {len(self.y_val)} 筆
- 測試集樣本數: {len(self.y_test)} 筆
- 分析模型數: {len(self.models)} 個

## 驗證集準確率排名

| 排名 | 模型 | 驗證集準確率 | 測試集準確率 | 準確率差異 |
|------|------|-------------|-------------|-----------|
"""

        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            report += f"| {i} | {row['Model']} | {row['Validation_Accuracy']:.4f} | {row['Test_Accuracy']:.4f} | {row['Accuracy_Diff']:.4f} |\n"

        report += f"""
## 最佳模型: {best_model['Model']}

### 性能指標
- 驗證集準確率: {best_model['Validation_Accuracy']:.4f} ({best_model['Validation_Accuracy']*100:.2f}%)
- 測試集準確率: {best_model['Test_Accuracy']:.4f} ({best_model['Test_Accuracy']*100:.2f}%)
- 準確率差異: {best_model['Accuracy_Diff']:.4f}

## 模型類型分析

### Baseline 模型
"""

        baseline_models = comparison_df[comparison_df["Model"].str.contains("Baseline")]
        for _, row in baseline_models.iterrows():
            report += f"- {row['Model']}: {row['Validation_Accuracy']:.4f} ({row['Validation_Accuracy']*100:.2f}%)\n"

        report += f"""
### 特徵選擇後模型
"""

        selected_models = comparison_df[comparison_df["Model"].str.contains("Selected")]
        for _, row in selected_models.iterrows():
            report += f"- {row['Model']}: {row['Validation_Accuracy']:.4f} ({row['Validation_Accuracy']*100:.2f}%)\n"

        report += f"""
## 結論

1. **最佳模型**: {best_model['Model']} 在驗證集上表現最佳
2. **模型穩定性**: 驗證集與測試集準確率差異分析
3. **特徵選擇效果**: 比較 Baseline 和特徵選擇後模型的性能
4. **模型選擇建議**: 基於驗證集性能選擇最佳模型

## 建議

1. **模型部署**: 使用 {best_model['Model']} 作為最終模型
2. **性能監控**: 持續監控模型在生產環境中的表現
3. **模型優化**: 基於驗證集結果進一步優化模型參數
4. **集成學習**: 考慮結合多個表現良好的模型
"""

        with open("results/validation_accuracy_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("驗證報告已儲存至 results/validation_accuracy_report.md")
        return report


def main():
    """主執行函數"""
    print("WineQT 驗證集準確率分析開始...")

    # 初始化分析器
    analyzer = ValidationAccuracyAnalyzer()

    # 載入資料
    analyzer.load_data()

    # 載入模型
    analyzer.load_models()

    # 評估驗證集準確率
    analyzer.evaluate_validation_accuracy()

    # 比較模型
    comparison_df = analyzer.compare_models()

    # 分析驗證集性能
    analyzer.analyze_validation_performance()

    # 視覺化結果
    analyzer.visualize_results(comparison_df)

    # 生成報告
    analyzer.generate_validation_report(comparison_df)

    print("\n驗證集準確率分析完成！")
    return analyzer, comparison_df


if __name__ == "__main__":
    analyzer, comparison_df = main()
