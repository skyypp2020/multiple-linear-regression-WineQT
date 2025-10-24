#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基於特徵重要性篩選的重新建模
只保留重要性在 0.03 以上的特徵
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class FeatureSelectionModeler:
    """基於特徵選擇的建模器"""

    def __init__(self, importance_threshold=0.03):
        """初始化建模器"""
        self.importance_threshold = importance_threshold
        self.selected_features = None
        self.models = {}
        self.results = {}
        self.baseline_results = {}

    def load_data_and_features(self):
        """載入資料和特徵重要性"""
        print("=== 載入資料和特徵重要性 ===")

        # 載入資料
        self.X_train = pd.read_csv("processed_data/X_train.csv")
        self.X_val = pd.read_csv("processed_data/X_val.csv")
        self.X_test = pd.read_csv("processed_data/X_test.csv")
        self.y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
        self.y_val = pd.read_csv("processed_data/y_val.csv").squeeze()
        self.y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

        # 載入特徵重要性
        self.feature_importance_df = pd.read_csv(
            "results/data/all_features_importance.csv"
        )

        print(f"原始特徵數量: {self.X_train.shape[1]}")
        print(f"特徵重要性閾值: {self.importance_threshold}")

        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )

    def select_features(self):
        """根據重要性閾值選擇特徵"""
        print(f"\n=== 特徵選擇 (閾值: {self.importance_threshold}) ===")

        # 選擇重要性大於閾值的特徵
        selected_mask = (
            self.feature_importance_df["Average_Importance"]
            >= self.importance_threshold
        )
        self.selected_features = self.feature_importance_df[selected_mask][
            "Feature"
        ].tolist()

        print(f"選中的特徵數量: {len(self.selected_features)}")
        print("選中的特徵:")
        for i, feature in enumerate(self.selected_features, 1):
            importance = self.feature_importance_df[
                self.feature_importance_df["Feature"] == feature
            ]["Average_Importance"].iloc[0]
            print(f"{i:2d}. {feature:25s}: {importance:.4f}")

        # 篩選特徵
        self.X_train_selected = self.X_train[self.selected_features]
        self.X_val_selected = self.X_val[self.selected_features]
        self.X_test_selected = self.X_test[self.selected_features]

        print(f"\n篩選後資料形狀:")
        print(f"訓練集: {self.X_train_selected.shape}")
        print(f"驗證集: {self.X_val_selected.shape}")
        print(f"測試集: {self.X_test_selected.shape}")

        return self.X_train_selected, self.X_val_selected, self.X_test_selected

    def create_models(self):
        """建立模型"""
        print("\n=== 建立特徵選擇後的模型 ===")

        # 使用相同的模型配置
        self.models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=42),
            "Lasso": Lasso(alpha=0.1, random_state=42),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        }

        print("已建立 4 個特徵選擇後的模型:")
        for name, model in self.models.items():
            print(f"- {name}")

        return self.models

    def train_models(self):
        """訓練模型"""
        print("\n=== 訓練特徵選擇後的模型 ===")

        for name, model in self.models.items():
            print(f"訓練 {name}...")
            model.fit(self.X_train_selected, self.y_train)
            print(f"  {name} 訓練完成")

        print("所有模型訓練完成！")
        return self.models

    def evaluate_models(self):
        """評估模型性能"""
        print("\n=== 特徵選擇後模型評估 ===")

        for name, model in self.models.items():
            print(f"\n{name} 模型評估:")

            # 預測
            y_train_pred = model.predict(self.X_train_selected)
            y_val_pred = model.predict(self.X_val_selected)
            y_test_pred = model.predict(self.X_test_selected)

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

            # 計算準確率 (四捨五入)
            y_test_pred_rounded = np.round(y_test_pred).astype(int)
            y_test_pred_rounded = np.clip(y_test_pred_rounded, 3, 8)
            test_accuracy = accuracy_score(self.y_test, y_test_pred_rounded)

            # 交叉驗證
            cv_scores = cross_val_score(
                model, self.X_train_selected, self.y_train, cv=5, scoring="r2"
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
                "test_accuracy": test_accuracy,
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
            print(f"  測試集準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"  交叉驗證 R2: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  測試集 RMSE: {test_rmse:.4f}")
            print(f"  測試集 MAE: {test_mae:.4f}")

        return self.results

    def load_baseline_results(self):
        """載入 Baseline 模型結果"""
        print("\n=== 載入 Baseline 模型結果 ===")

        # 載入 Baseline 模型
        baseline_models = {
            "LinearRegression": joblib.load(
                "results/models/linearregression_model.pkl"
            ),
            "Ridge": joblib.load("results/models/ridge_model.pkl"),
            "Lasso": joblib.load("results/models/lasso_model.pkl"),
            "ElasticNet": joblib.load("results/models/elasticnet_model.pkl"),
        }

        # 評估 Baseline 模型
        for name, model in baseline_models.items():
            y_test_pred = model.predict(self.X_test)
            y_test_pred_rounded = np.round(y_test_pred).astype(int)
            y_test_pred_rounded = np.clip(y_test_pred_rounded, 3, 8)
            test_accuracy = accuracy_score(self.y_test, y_test_pred_rounded)

            test_r2 = r2_score(self.y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            test_mae = mean_absolute_error(self.y_test, y_test_pred)

            self.baseline_results[name] = {
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_accuracy": test_accuracy,
            }

        print("Baseline 模型結果載入完成")
        return self.baseline_results

    def compare_with_baseline(self):
        """與 Baseline 模型比較"""
        print("\n=== 與 Baseline 模型比較 ===")

        # 創建比較表格
        comparison_data = []

        for name in self.models.keys():
            # 特徵選擇後模型結果
            selected_result = self.results[name]
            # Baseline 模型結果
            baseline_result = self.baseline_results[name]

            comparison_data.append(
                {
                    "Model": name,
                    "Feature_Count": len(self.selected_features),
                    "Selected_Accuracy": selected_result["test_accuracy"],
                    "Baseline_Accuracy": baseline_result["test_accuracy"],
                    "Accuracy_Improvement": selected_result["test_accuracy"]
                    - baseline_result["test_accuracy"],
                    "Selected_R2": selected_result["test_r2"],
                    "Baseline_R2": baseline_result["test_r2"],
                    "R2_Improvement": selected_result["test_r2"]
                    - baseline_result["test_r2"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Selected_Accuracy", ascending=False)

        print("模型性能比較:")
        print("=" * 100)
        print(
            f"{'模型':<15} {'特徵數':<8} {'選擇後準確率':<12} {'Baseline準確率':<15} {'準確率提升':<10} {'R2提升':<10}"
        )
        print("-" * 100)

        for _, row in comparison_df.iterrows():
            print(
                f"{row['Model']:<15} {row['Feature_Count']:<8} "
                f"{row['Selected_Accuracy']:<12.4f} {row['Baseline_Accuracy']:<15.4f} "
                f"{row['Accuracy_Improvement']:<10.4f} {row['R2_Improvement']:<10.4f}"
            )

        # 找出最佳改進模型
        best_improvement = comparison_df.loc[
            comparison_df["Accuracy_Improvement"].idxmax()
        ]
        print(f"\n最佳改進模型: {best_improvement['Model']}")
        print(
            f"準確率提升: {best_improvement['Accuracy_Improvement']:.4f} ({best_improvement['Accuracy_Improvement']*100:.2f}%)"
        )
        print(f"R2 提升: {best_improvement['R2_Improvement']:.4f}")

        return comparison_df

    def visualize_comparison(self, comparison_df):
        """視覺化比較結果"""
        print("\n=== 生成比較視覺化圖表 ===")

        # 創建結果目錄
        os.makedirs("results/plots", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 準確率比較
        models = comparison_df["Model"]
        selected_acc = comparison_df["Selected_Accuracy"]
        baseline_acc = comparison_df["Baseline_Accuracy"]

        x = np.arange(len(models))
        width = 0.35

        axes[0, 0].bar(
            x - width / 2,
            baseline_acc,
            width,
            label="Baseline",
            alpha=0.8,
            color="skyblue",
        )
        axes[0, 0].bar(
            x + width / 2,
            selected_acc,
            width,
            label="特徵選擇後",
            alpha=0.8,
            color="lightcoral",
        )
        axes[0, 0].set_xlabel("模型")
        axes[0, 0].set_ylabel("準確率")
        axes[0, 0].set_title("準確率比較")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 準確率提升
        improvement = comparison_df["Accuracy_Improvement"]
        colors = ["green" if x > 0 else "red" for x in improvement]
        axes[0, 1].bar(models, improvement, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel("模型")
        axes[0, 1].set_ylabel("準確率提升")
        axes[0, 1].set_title("準確率提升比較")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. R2 比較
        selected_r2 = comparison_df["Selected_R2"]
        baseline_r2 = comparison_df["Baseline_R2"]

        axes[1, 0].bar(
            x - width / 2,
            baseline_r2,
            width,
            label="Baseline",
            alpha=0.8,
            color="skyblue",
        )
        axes[1, 0].bar(
            x + width / 2,
            selected_r2,
            width,
            label="特徵選擇後",
            alpha=0.8,
            color="lightcoral",
        )
        axes[1, 0].set_xlabel("模型")
        axes[1, 0].set_ylabel("R2 Score")
        axes[1, 0].set_title("R2 比較")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 特徵數量 vs 準確率
        feature_count = comparison_df["Feature_Count"]
        axes[1, 1].scatter(
            feature_count,
            selected_acc,
            s=100,
            alpha=0.7,
            color="red",
            label="特徵選擇後",
        )
        axes[1, 1].scatter(
            [15] * len(models),
            baseline_acc,
            s=100,
            alpha=0.7,
            color="blue",
            label="Baseline (15特徵)",
        )
        axes[1, 1].set_xlabel("特徵數量")
        axes[1, 1].set_ylabel("準確率")
        axes[1, 1].set_title("特徵數量 vs 準確率")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 添加模型標籤
        for i, model in enumerate(models):
            axes[1, 1].annotate(
                model,
                (feature_count.iloc[i], selected_acc.iloc[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(
            "results/plots/feature_selection_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print("比較視覺化圖表已儲存至 results/plots/feature_selection_comparison.png")

    def save_results(self, comparison_df):
        """儲存結果"""
        print("\n=== 儲存特徵選擇結果 ===")

        # 創建結果目錄
        os.makedirs("results/models/feature_selected", exist_ok=True)
        os.makedirs("results/data", exist_ok=True)

        # 儲存特徵選擇後的模型
        for name, model in self.models.items():
            joblib.dump(
                model,
                f"results/models/feature_selected/{name.lower()}_selected_model.pkl",
            )

        # 儲存比較結果
        comparison_df.to_csv(
            "results/data/feature_selection_comparison.csv", index=False
        )

        # 儲存選中的特徵
        selected_features_df = pd.DataFrame(
            {
                "Feature": self.selected_features,
                "Importance": [
                    self.feature_importance_df[
                        self.feature_importance_df["Feature"] == f
                    ]["Average_Importance"].iloc[0]
                    for f in self.selected_features
                ],
            }
        )
        selected_features_df.to_csv("results/data/selected_features.csv", index=False)

        print("結果已儲存至 results/ 目錄")
        print("檔案列表:")
        print("- results/models/feature_selected/: 特徵選擇後的模型")
        print("- results/data/feature_selection_comparison.csv: 比較結果")
        print("- results/data/selected_features.csv: 選中的特徵")

    def generate_report(self, comparison_df):
        """生成特徵選擇報告"""
        print("\n=== 生成特徵選擇報告 ===")

        # 找出最佳改進模型
        best_model = comparison_df.loc[comparison_df["Accuracy_Improvement"].idxmax()]

        report = f"""
# WineQT 特徵選擇建模報告

## 概述
- 分析時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 特徵重要性閾值: {self.importance_threshold}
- 原始特徵數量: 15 個
- 選中特徵數量: {len(self.selected_features)} 個
- 特徵減少比例: {(15 - len(self.selected_features)) / 15 * 100:.1f}%

## 選中的特徵
"""

        for i, feature in enumerate(self.selected_features, 1):
            importance = self.feature_importance_df[
                self.feature_importance_df["Feature"] == feature
            ]["Average_Importance"].iloc[0]
            report += f"{i}. {feature}: {importance:.4f}\n"

        report += f"""
## 模型性能比較

| 模型 | 特徵數 | 選擇後準確率 | Baseline準確率 | 準確率提升 | R2提升 |
|------|--------|-------------|---------------|-----------|--------|
"""

        for _, row in comparison_df.iterrows():
            report += f"| {row['Model']} | {row['Feature_Count']} | {row['Selected_Accuracy']:.4f} | {row['Baseline_Accuracy']:.4f} | {row['Accuracy_Improvement']:.4f} | {row['R2_Improvement']:.4f} |\n"

        report += f"""
## 最佳改進模型: {best_model['Model']}

### 性能提升
- 準確率提升: {best_model['Accuracy_Improvement']:.4f} ({best_model['Accuracy_Improvement']*100:.2f}%)
- R2 提升: {best_model['R2_Improvement']:.4f}
- 特徵數量: {best_model['Feature_Count']} 個 (減少 {15 - best_model['Feature_Count']} 個)

## 結論
"""

        if best_model["Accuracy_Improvement"] > 0:
            report += f"✅ 特徵選擇成功提升了模型性能，{best_model['Model']} 模型準確率提升了 {best_model['Accuracy_Improvement']*100:.2f}%"
        else:
            report += f"❌ 特徵選擇未能提升模型性能，建議調整特徵選擇策略"

        report += f"""

## 建議
1. **特徵選擇效果**: 使用重要性閾值 {self.importance_threshold} 成功減少了 {15 - len(self.selected_features)} 個特徵
2. **模型優化**: 可以嘗試不同的重要性閾值進行特徵選擇
3. **進一步改進**: 考慮使用更進階的特徵選擇方法 (如 RFE, SelectKBest)
4. **集成學習**: 結合多個特徵選擇後的模型進行集成
"""

        with open("results/feature_selection_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("特徵選擇報告已儲存至 results/feature_selection_report.md")
        return report


def main():
    """主執行函數"""
    print("WineQT 特徵選擇建模開始...")

    # 初始化建模器
    modeler = FeatureSelectionModeler(importance_threshold=0.03)

    # 載入資料和特徵重要性
    modeler.load_data_and_features()

    # 選擇特徵
    modeler.select_features()

    # 建立模型
    modeler.create_models()

    # 訓練模型
    modeler.train_models()

    # 評估模型
    modeler.evaluate_models()

    # 載入 Baseline 結果
    modeler.load_baseline_results()

    # 比較結果
    comparison_df = modeler.compare_with_baseline()

    # 視覺化
    modeler.visualize_comparison(comparison_df)

    # 儲存結果
    modeler.save_results(comparison_df)

    # 生成報告
    modeler.generate_report(comparison_df)

    print("\n特徵選擇建模完成！")
    return modeler, comparison_df


if __name__ == "__main__":
    modeler, comparison_df = main()
