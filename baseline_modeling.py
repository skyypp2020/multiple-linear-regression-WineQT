#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WineQT 多元線性迴歸 Baseline 建模
基於 CRISP-DM 架構的建模階段
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class WineBaselineModeler:
    """WineQT 多元線性迴歸 Baseline 建模類別"""

    def __init__(self):
        """初始化建模器"""
        self.models = {}
        self.results = {}
        self.feature_importance = {}

    def load_processed_data(self):
        """載入處理後的資料"""
        print("=== 載入處理後資料 ===")

        # 載入特徵資料
        self.X_train = pd.read_csv("processed_data/X_train.csv")
        self.X_val = pd.read_csv("processed_data/X_val.csv")
        self.X_test = pd.read_csv("processed_data/X_test.csv")

        # 載入目標變數
        self.y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
        self.y_val = pd.read_csv("processed_data/y_val.csv").squeeze()
        self.y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

        print(f"訓練集: {self.X_train.shape[0]} 筆 x {self.X_train.shape[1]} 特徵")
        print(f"驗證集: {self.X_val.shape[0]} 筆 x {self.X_val.shape[1]} 特徵")
        print(f"測試集: {self.X_test.shape[0]} 筆 x {self.X_test.shape[1]} 特徵")

        # 顯示特徵名稱
        print(f"\n特徵列表 ({len(self.X_train.columns)} 個):")
        for i, feature in enumerate(self.X_train.columns, 1):
            print(f"{i:2d}. {feature}")

        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )

    def create_baseline_models(self):
        """建立 Baseline 模型"""
        print("\n=== 建立 Baseline 模型 ===")

        # 1. 多元線性迴歸 (基礎模型)
        self.models["LinearRegression"] = LinearRegression()

        # 2. 嶺迴歸 (Ridge Regression)
        self.models["Ridge"] = Ridge(alpha=1.0, random_state=42)

        # 3. 套索迴歸 (Lasso Regression)
        self.models["Lasso"] = Lasso(alpha=0.1, random_state=42)

        # 4. 彈性網路 (Elastic Net)
        self.models["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

        print("已建立 4 個 Baseline 模型:")
        for name, model in self.models.items():
            print(f"- {name}")

        return self.models

    def train_models(self):
        """訓練所有模型"""
        print("\n=== 訓練模型 ===")

        for name, model in self.models.items():
            print(f"訓練 {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"  {name} 訓練完成")

        print("所有模型訓練完成！")
        return self.models

    def evaluate_models(self):
        """評估模型性能"""
        print("\n=== 模型評估 ===")

        evaluation_metrics = ["R²", "RMSE", "MAE"]

        for name, model in self.models.items():
            print(f"\n{name} 模型評估:")

            # 預測
            y_train_pred = model.predict(self.X_train)
            y_val_pred = model.predict(self.X_val)
            y_test_pred = model.predict(self.X_test)

            # 計算評估指標
            train_r2 = r2_score(self.y_train, y_train_pred)
            val_r2 = r2_score(self.y_val, y_val_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)

            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, y_val_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            val_mae = mean_absolute_error(self.y_val, y_val_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)

            # 交叉驗證
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, cv=5, scoring="r2"
            )

            # 儲存結果
            self.results[name] = {
                "train_r2": train_r2,
                "val_r2": val_r2,
                "test_r2": test_r2,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "predictions": {
                    "train": y_train_pred,
                    "val": y_val_pred,
                    "test": y_test_pred,
                },
            }

            # 顯示結果
            print(f"  訓練集 R2: {train_r2:.4f}")
            print(f"  驗證集 R2: {val_r2:.4f}")
            print(f"  測試集 R2: {test_r2:.4f}")
            print(f"  交叉驗證 R2: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  測試集 RMSE: {test_rmse:.4f}")
            print(f"  測試集 MAE: {test_mae:.4f}")

        return self.results

    def analyze_feature_importance(self):
        """分析特徵重要性"""
        print("\n=== 特徵重要性分析 ===")

        for name, model in self.models.items():
            if hasattr(model, "coef_"):
                # 線性模型的係數
                coefficients = model.coef_
                feature_names = self.X_train.columns

                # 創建特徵重要性 DataFrame
                importance_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "coefficient": coefficients,
                        "abs_coefficient": np.abs(coefficients),
                    }
                ).sort_values("abs_coefficient", ascending=False)

                self.feature_importance[name] = importance_df

                print(f"\n{name} 特徵重要性 (前 10 名):")
                top_features = importance_df.head(10)
                for idx, row in top_features.iterrows():
                    print(f"  {row['feature']:25s}: {row['coefficient']:8.4f}")

        return self.feature_importance

    def compare_models(self):
        """比較模型性能"""
        print("\n=== 模型比較 ===")

        # 創建比較表格
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append(
                {
                    "Model": name,
                    "Test_R2": results["test_r2"],
                    "Test_RMSE": results["test_rmse"],
                    "Test_MAE": results["test_mae"],
                    "CV_R2_Mean": results["cv_mean"],
                    "CV_R2_Std": results["cv_std"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Test_R2", ascending=False)

        print("模型性能比較 (按測試集 R2 排序):")
        print(comparison_df.to_string(index=False, float_format="%.4f"))

        # 找出最佳模型
        best_model_name = comparison_df.iloc[0]["Model"]
        best_r2 = comparison_df.iloc[0]["Test_R2"]

        print(f"\n最佳模型: {best_model_name}")
        print(f"最佳 R2: {best_r2:.4f}")

        return comparison_df, best_model_name

    def visualize_results(self):
        """視覺化結果"""
        print("\n=== 生成視覺化圖表 ===")

        # 創建結果目錄
        os.makedirs("results/plots", exist_ok=True)

        # 1. 模型性能比較圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # R² 比較
        models = list(self.results.keys())
        test_r2_scores = [self.results[model]["test_r2"] for model in models]
        val_r2_scores = [self.results[model]["val_r2"] for model in models]

        axes[0, 0].bar(models, test_r2_scores, alpha=0.7, color="skyblue")
        axes[0, 0].set_title("測試集 R2 比較")
        axes[0, 0].set_ylabel("R2 Score")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # RMSE 比較
        test_rmse_scores = [self.results[model]["test_rmse"] for model in models]
        axes[0, 1].bar(models, test_rmse_scores, alpha=0.7, color="lightcoral")
        axes[0, 1].set_title("測試集 RMSE 比較")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 實際值 vs 預測值 (最佳模型)
        best_model_name = max(
            self.results.keys(), key=lambda x: self.results[x]["test_r2"]
        )
        y_test_pred = self.results[best_model_name]["predictions"]["test"]

        axes[1, 0].scatter(self.y_test, y_test_pred, alpha=0.6)
        axes[1, 0].plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--",
            lw=2,
        )
        axes[1, 0].set_xlabel("實際值")
        axes[1, 0].set_ylabel("預測值")
        axes[1, 0].set_title(f"{best_model_name} - 實際值 vs 預測值")

        # 殘差圖
        residuals = self.y_test - y_test_pred
        axes[1, 1].scatter(y_test_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color="r", linestyle="--")
        axes[1, 1].set_xlabel("預測值")
        axes[1, 1].set_ylabel("殘差")
        axes[1, 1].set_title(f"{best_model_name} - 殘差圖")

        plt.tight_layout()
        plt.savefig(
            "results/plots/baseline_model_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # 2. 特徵重要性圖 (最佳模型)
        if best_model_name in self.feature_importance:
            importance_df = self.feature_importance[best_model_name]
            top_features = importance_df.head(15)

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_features["coefficient"])
            plt.yticks(range(len(top_features)), top_features["feature"])
            plt.xlabel("係數值")
            plt.title(f"{best_model_name} - 特徵重要性 (前 15 名)")
            plt.tight_layout()
            plt.savefig(
                "results/plots/feature_importance.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

        print("視覺化圖表已儲存至 results/plots/ 目錄")

    def save_models_and_results(self):
        """儲存模型和結果"""
        print("\n=== 儲存模型和結果 ===")

        # 創建結果目錄
        os.makedirs("results/models", exist_ok=True)
        os.makedirs("results/data", exist_ok=True)

        # 儲存模型
        for name, model in self.models.items():
            joblib.dump(model, f"results/models/{name.lower()}_model.pkl")

        # 儲存結果
        results_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Test_R²": results["test_r2"],
                    "Test_RMSE": results["test_rmse"],
                    "Test_MAE": results["test_mae"],
                    "CV_R2_Mean": results["cv_mean"],
                    "CV_R2_Std": results["cv_std"],
                }
                for name, results in self.results.items()
            ]
        )
        results_df.to_csv("results/data/baseline_model_results.csv", index=False)

        # 儲存特徵重要性
        for name, importance_df in self.feature_importance.items():
            importance_df.to_csv(
                f"results/data/{name.lower()}_feature_importance.csv", index=False
            )

        print("模型和結果已儲存至 results/ 目錄")
        print("檔案列表:")
        print("- results/models/: 訓練好的模型檔案")
        print("- results/data/: 評估結果和特徵重要性")
        print("- results/plots/: 視覺化圖表")

    def generate_model_report(self):
        """生成模型報告"""
        print("\n=== 生成模型報告 ===")

        # 找出最佳模型
        best_model_name = max(
            self.results.keys(), key=lambda x: self.results[x]["test_r2"]
        )
        best_results = self.results[best_model_name]

        report = f"""
# WineQT 多元線性迴歸 Baseline 模型報告

## 模型概述
- 建立時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 資料集: WineQT 處理後資料
- 特徵數量: {self.X_train.shape[1]} 個
- 訓練樣本: {self.X_train.shape[0]} 筆
- 驗證樣本: {self.X_val.shape[0]} 筆
- 測試樣本: {self.X_test.shape[0]} 筆

## 模型性能比較

| 模型 | 測試集 R2 | 測試集 RMSE | 測試集 MAE | 交叉驗證 R2 |
|------|-----------|-------------|------------|-------------|
"""

        for name, results in self.results.items():
            report += f"| {name} | {results['test_r2']:.4f} | {results['test_rmse']:.4f} | {results['test_mae']:.4f} | {results['cv_mean']:.4f} |\n"

        report += f"""
## 最佳模型: {best_model_name}

### 性能指標
- 測試集 R2: {best_results['test_r2']:.4f}
- 測試集 RMSE: {best_results['test_rmse']:.4f}
- 測試集 MAE: {best_results['test_mae']:.4f}
- 交叉驗證 R2: {best_results['cv_mean']:.4f} (±{best_results['cv_std']:.4f})

### 特徵重要性 (前 10 名)
"""

        if best_model_name in self.feature_importance:
            top_features = self.feature_importance[best_model_name].head(10)
            for idx, row in top_features.iterrows():
                report += f"- {row['feature']}: {row['coefficient']:.4f}\n"

        report += """
## 模型解釋
- R2: 模型解釋的變異比例
- RMSE: 均方根誤差，單位與目標變數相同
- MAE: 平均絕對誤差，單位與目標變數相同
- 交叉驗證: 5折交叉驗證的平均性能

## 結論
Baseline 模型已建立完成，可作為後續模型改進的基準。
"""

        with open("results/baseline_model_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("模型報告已儲存至 results/baseline_model_report.md")
        return report


def main():
    """主執行函數"""
    print("WineQT 多元線性迴歸 Baseline 建模開始...")

    # 初始化建模器
    modeler = WineBaselineModeler()

    # 載入資料
    modeler.load_processed_data()

    # 建立模型
    modeler.create_baseline_models()

    # 訓練模型
    modeler.train_models()

    # 評估模型
    modeler.evaluate_models()

    # 分析特徵重要性
    modeler.analyze_feature_importance()

    # 比較模型
    comparison_df, best_model = modeler.compare_models()

    # 視覺化結果
    modeler.visualize_results()

    # 儲存模型和結果
    modeler.save_models_and_results()

    # 生成報告
    modeler.generate_model_report()

    print("\nBaseline 建模完成！")
    return modeler


if __name__ == "__main__":
    modeler = main()
