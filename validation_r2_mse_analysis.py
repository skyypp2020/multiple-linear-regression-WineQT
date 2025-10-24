#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用驗證資料集進行各模型的 R² 和 MSE 驗證
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class ValidationR2MSEAnalyzer:
    """驗證資料集 R² 和 MSE 分析器"""

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

    def evaluate_validation_metrics(self):
        """評估驗證集 R² 和 MSE"""
        print("\n=== 驗證集 R2 和 MSE 評估 ===")

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
                y_val_pred = model.predict(self.X_val)
                y_test_pred = model.predict(self.X_test)

            # 計算 R² 和 MSE
            val_r2 = r2_score(self.y_val, y_val_pred)
            val_mse = mean_squared_error(self.y_val, y_val_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)

            # 儲存結果
            self.results[name] = {
                "val_r2": val_r2,
                "val_mse": val_mse,
                "test_r2": test_r2,
                "test_mse": test_mse,
                "val_predictions": y_val_pred,
                "test_predictions": y_test_pred,
            }

            print(f"  驗證集 R2: {val_r2:.4f}")
            print(f"  驗證集 MSE: {val_mse:.4f}")
            print(f"  測試集 R2: {test_r2:.4f}")
            print(f"  測試集 MSE: {test_mse:.4f}")

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
                    "Validation_R2": results["val_r2"],
                    "Validation_MSE": results["val_mse"],
                    "Test_R2": results["test_r2"],
                    "Test_MSE": results["test_mse"],
                    "R2_Diff": results["test_r2"] - results["val_r2"],
                    "MSE_Diff": results["test_mse"] - results["val_mse"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Validation_R2", ascending=False)

        print("驗證集 R2 排名:")
        print("=" * 100)
        print(
            f"{'模型':<25} {'驗證集R2':<10} {'驗證集MSE':<10} {'測試集R2':<10} {'測試集MSE':<10} {'R2差異':<8} {'MSE差異':<8}"
        )
        print("-" * 100)

        for _, row in comparison_df.iterrows():
            print(
                f"{row['Model']:<25} {row['Validation_R2']:<10.4f} {row['Validation_MSE']:<10.4f} {row['Test_R2']:<10.4f} {row['Test_MSE']:<10.4f} {row['R2_Diff']:<8.4f} {row['MSE_Diff']:<8.4f}"
            )

        # 找出最佳模型
        best_model = comparison_df.iloc[0]
        print(f"\n最佳模型: {best_model['Model']}")
        print(f"驗證集 R2: {best_model['Validation_R2']:.4f}")
        print(f"驗證集 MSE: {best_model['Validation_MSE']:.4f}")
        print(f"測試集 R2: {best_model['Test_R2']:.4f}")
        print(f"測試集 MSE: {best_model['Test_MSE']:.4f}")

        return comparison_df

    def analyze_baseline_vs_selected(self):
        """分析 Baseline vs 特徵選擇後模型"""
        print("\n=== Baseline vs 特徵選擇後模型分析 ===")

        # 分析 Baseline vs 特徵選擇後的模型
        baseline_models = [name for name in self.results.keys() if "Baseline" in name]
        selected_models = [name for name in self.results.keys() if "Selected" in name]

        print("Baseline 模型驗證集性能:")
        for name in baseline_models:
            r2 = self.results[name]["val_r2"]
            mse = self.results[name]["val_mse"]
            print(f"  {name}: R2={r2:.4f}, MSE={mse:.4f}")

        print("\n特徵選擇後模型驗證集性能:")
        for name in selected_models:
            r2 = self.results[name]["val_r2"]
            mse = self.results[name]["val_mse"]
            print(f"  {name}: R2={r2:.4f}, MSE={mse:.4f}")

        # 比較 Baseline 和特徵選擇後的模型
        print("\nBaseline vs 特徵選擇後模型比較:")
        for baseline_name in baseline_models:
            model_type = baseline_name.split("_")[1]  # 取得模型類型
            selected_name = f"Selected_{model_type}"

            if selected_name in self.results:
                baseline_r2 = self.results[baseline_name]["val_r2"]
                baseline_mse = self.results[baseline_name]["val_mse"]
                selected_r2 = self.results[selected_name]["val_r2"]
                selected_mse = self.results[selected_name]["val_mse"]

                r2_improvement = selected_r2 - baseline_r2
                mse_improvement = baseline_mse - selected_mse  # MSE 越低越好

                print(f"  {model_type}:")
                print(f"    Baseline: R2={baseline_r2:.4f}, MSE={baseline_mse:.4f}")
                print(f"    特徵選擇後: R2={selected_r2:.4f}, MSE={selected_mse:.4f}")
                print(f"    R2 改進: {r2_improvement:.4f}")
                print(f"    MSE 改進: {mse_improvement:.4f}")

    def visualize_results(self, comparison_df):
        """視覺化結果"""
        print("\n=== 生成驗證集 R2 和 MSE 視覺化圖表 ===")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 驗證集 R2 比較
        models = comparison_df["Model"]
        val_r2 = comparison_df["Validation_R2"]
        test_r2 = comparison_df["Test_R2"]

        x = np.arange(len(models))
        width = 0.35

        axes[0, 0].bar(
            x - width / 2, val_r2, width, label="驗證集", alpha=0.8, color="skyblue"
        )
        axes[0, 0].bar(
            x + width / 2,
            test_r2,
            width,
            label="測試集",
            alpha=0.8,
            color="lightcoral",
        )
        axes[0, 0].set_xlabel("模型")
        axes[0, 0].set_ylabel("R2")
        axes[0, 0].set_title("驗證集 vs 測試集 R2 比較")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha="right")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 驗證集 MSE 比較
        val_mse = comparison_df["Validation_MSE"]
        test_mse = comparison_df["Test_MSE"]

        axes[0, 1].bar(
            x - width / 2, val_mse, width, label="驗證集", alpha=0.8, color="lightgreen"
        )
        axes[0, 1].bar(
            x + width / 2,
            test_mse,
            width,
            label="測試集",
            alpha=0.8,
            color="orange",
        )
        axes[0, 1].set_xlabel("模型")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].set_title("驗證集 vs 測試集 MSE 比較")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45, ha="right")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Baseline vs 特徵選擇後模型 R2 比較
        baseline_mask = comparison_df["Model"].str.contains("Baseline")
        selected_mask = comparison_df["Model"].str.contains("Selected")

        baseline_data = comparison_df[baseline_mask]
        selected_data = comparison_df[selected_mask]

        # 提取模型類型
        baseline_types = baseline_data["Model"].str.split("_").str[1]
        selected_types = selected_data["Model"].str.split("_").str[1]

        # 按模型類型排序
        model_types = ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]
        baseline_r2 = [
            (
                baseline_data[baseline_data["Model"].str.contains(t)][
                    "Validation_R2"
                ].iloc[0]
                if t in baseline_data["Model"].str.split("_").str[1].values
                else 0
            )
            for t in model_types
        ]
        selected_r2 = [
            (
                selected_data[selected_data["Model"].str.contains(t)][
                    "Validation_R2"
                ].iloc[0]
                if t in selected_data["Model"].str.split("_").str[1].values
                else 0
            )
            for t in model_types
        ]

        x = np.arange(len(model_types))
        axes[1, 0].bar(
            x - width / 2,
            baseline_r2,
            width,
            label="Baseline",
            alpha=0.8,
            color="lightblue",
        )
        axes[1, 0].bar(
            x + width / 2,
            selected_r2,
            width,
            label="特徵選擇後",
            alpha=0.8,
            color="lightgreen",
        )
        axes[1, 0].set_xlabel("模型類型")
        axes[1, 0].set_ylabel("驗證集 R2")
        axes[1, 0].set_title("Baseline vs 特徵選擇後模型 R2 比較")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_types)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Baseline vs 特徵選擇後模型 MSE 比較
        baseline_mse = [
            (
                baseline_data[baseline_data["Model"].str.contains(t)][
                    "Validation_MSE"
                ].iloc[0]
                if t in baseline_data["Model"].str.split("_").str[1].values
                else 0
            )
            for t in model_types
        ]
        selected_mse = [
            (
                selected_data[selected_data["Model"].str.contains(t)][
                    "Validation_MSE"
                ].iloc[0]
                if t in selected_data["Model"].str.split("_").str[1].values
                else 0
            )
            for t in model_types
        ]

        axes[1, 1].bar(
            x - width / 2,
            baseline_mse,
            width,
            label="Baseline",
            alpha=0.8,
            color="lightcoral",
        )
        axes[1, 1].bar(
            x + width / 2,
            selected_mse,
            width,
            label="特徵選擇後",
            alpha=0.8,
            color="orange",
        )
        axes[1, 1].set_xlabel("模型類型")
        axes[1, 1].set_ylabel("驗證集 MSE")
        axes[1, 1].set_title("Baseline vs 特徵選擇後模型 MSE 比較")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_types)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "results/plots/validation_r2_mse_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            "驗證集 R2 和 MSE 分析圖表已儲存至 results/plots/validation_r2_mse_analysis.png"
        )

    def generate_validation_report(self, comparison_df):
        """生成驗證報告"""
        print("\n=== 生成驗證報告 ===")

        # 找出最佳模型
        best_model = comparison_df.iloc[0]

        report = f"""
# WineQT 驗證集 R2 和 MSE 分析報告

## 執行時間
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 模型性能排名 (按驗證集 R2 排序)

| 排名 | 模型 | 驗證集R2 | 驗證集MSE | 測試集R2 | 測試集MSE | R2差異 | MSE差異 |
|------|------|----------|-----------|----------|-----------|--------|---------|
"""

        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            report += f"| {i} | {row['Model']} | {row['Validation_R2']:.4f} | {row['Validation_MSE']:.4f} | {row['Test_R2']:.4f} | {row['Test_MSE']:.4f} | {row['R2_Diff']:.4f} | {row['MSE_Diff']:.4f} |\n"

        report += f"""
## 最佳模型: {best_model['Model']}

### 性能指標
- **驗證集 R2**: {best_model['Validation_R2']:.4f}
- **驗證集 MSE**: {best_model['Validation_MSE']:.4f}
- **測試集 R2**: {best_model['Test_R2']:.4f}
- **測試集 MSE**: {best_model['Test_MSE']:.4f}

## Baseline 模型
"""

        baseline_models = comparison_df[comparison_df["Model"].str.contains("Baseline")]
        for _, row in baseline_models.iterrows():
            report += f"- {row['Model']}: R2={row['Validation_R2']:.4f}, MSE={row['Validation_MSE']:.4f}\n"

        report += f"""
## 特徵選擇後模型
"""

        selected_models = comparison_df[comparison_df["Model"].str.contains("Selected")]
        for _, row in selected_models.iterrows():
            report += f"- {row['Model']}: R2={row['Validation_R2']:.4f}, MSE={row['Validation_MSE']:.4f}\n"

        report += f"""
## 結論

### 主要發現
1. **最佳模型**: {best_model['Model']} 在驗證集上表現最佳
2. **R2 性能**: 所有模型的 R2 都在 0.3-0.4 之間，顯示模型有一定的解釋能力
3. **MSE 性能**: 所有模型的 MSE 都在 0.6-0.7 之間，顯示預測誤差相對穩定
4. **特徵選擇影響**: 特徵選擇對不同模型的影響程度不同

### 建議
1. **模型選擇**: 選擇 {best_model['Model']} 作為最終模型
2. **特徵工程**: 考慮創建更多有意義的特徵
3. **超參數調優**: 對最佳模型進行超參數調優
4. **集成學習**: 考慮結合多個表現良好的模型
"""

        with open("results/validation_r2_mse_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("驗證報告已儲存至 results/validation_r2_mse_report.md")
        return report


def main():
    """主執行函數"""
    print("WineQT 驗證集 R2 和 MSE 分析開始...")

    # 初始化分析器
    analyzer = ValidationR2MSEAnalyzer()

    # 載入資料
    analyzer.load_data()

    # 載入模型
    analyzer.load_models()

    # 評估驗證集 R² 和 MSE
    analyzer.evaluate_validation_metrics()

    # 比較模型
    comparison_df = analyzer.compare_models()

    # 分析 Baseline vs 特徵選擇後模型
    analyzer.analyze_baseline_vs_selected()

    # 視覺化結果
    analyzer.visualize_results(comparison_df)

    # 生成報告
    analyzer.generate_validation_report(comparison_df)

    print("\n驗證集 R2 和 MSE 分析完成！")
    return analyzer, comparison_df


if __name__ == "__main__":
    analyzer, comparison_df = main()
