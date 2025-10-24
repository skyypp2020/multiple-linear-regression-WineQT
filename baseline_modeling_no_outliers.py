#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WineQT 多元線性迴歸建模 (保留異常值版本)
建立 Baseline 模型，不新增特徵
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class WineModelingNoOutliers:
    """WineQT 多元線性迴歸建模類別 (保留異常值版本)"""

    def __init__(self):
        """初始化建模器"""
        self.models = {}
        self.results = {}
        self.feature_names = None

    def load_data(self):
        """載入資料"""
        print("=== 載入資料 ===")

        # 載入訓練資料
        self.X_train = pd.read_csv("processed_data_no_outliers/X_train.csv")
        self.y_train = pd.read_csv("processed_data_no_outliers/y_train.csv").squeeze()

        # 載入驗證資料
        self.X_val = pd.read_csv("processed_data_no_outliers/X_val.csv")
        self.y_val = pd.read_csv("processed_data_no_outliers/y_val.csv").squeeze()

        # 載入測試資料
        self.X_test = pd.read_csv("processed_data_no_outliers/X_test.csv")
        self.y_test = pd.read_csv("processed_data_no_outliers/y_test.csv").squeeze()

        self.feature_names = self.X_train.columns.tolist()

        print(f"訓練集: {self.X_train.shape[0]} 筆 x {self.X_train.shape[1]} 特徵")
        print(f"驗證集: {self.X_val.shape[0]} 筆 x {self.X_val.shape[1]} 特徵")
        print(f"測試集: {self.X_test.shape[0]} 筆 x {self.X_test.shape[1]} 特徵")
        print(f"特徵名稱: {self.feature_names}")

        return (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )

    def build_models(self):
        """建立模型"""
        print("\n=== 建立模型 ===")

        # 定義模型
        self.models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        }

        print("已建立 4 個模型:")
        for name, model in self.models.items():
            print(f"- {name}")

        return self.models

    def train_models(self):
        """訓練模型"""
        print("\n=== 訓練模型 ===")

        for name, model in self.models.items():
            print(f"\n訓練 {name} 模型...")

            # 訓練模型
            model.fit(self.X_train, self.y_train)

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
                "model": model,
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
                "y_train_pred": y_train_pred,
                "y_val_pred": y_val_pred,
                "y_test_pred": y_test_pred,
            }

            print(f"  訓練集 R2: {train_r2:.4f}")
            print(f"  驗證集 R2: {val_r2:.4f}")
            print(f"  測試集 R2: {test_r2:.4f}")
            print(f"  交叉驗證 R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

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
                    "Train_R2": results["train_r2"],
                    "Val_R2": results["val_r2"],
                    "Test_R2": results["test_r2"],
                    "Train_RMSE": results["train_rmse"],
                    "Val_RMSE": results["val_rmse"],
                    "Test_RMSE": results["test_rmse"],
                    "Train_MAE": results["train_mae"],
                    "Val_MAE": results["val_mae"],
                    "Test_MAE": results["test_mae"],
                    "CV_Mean": results["cv_mean"],
                    "CV_Std": results["cv_std"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Test_R2", ascending=False)

        print("模型性能排名 (按測試集 R2 排序):")
        print("=" * 100)
        print(
            f"{'模型':<15} {'訓練R2':<8} {'驗證R2':<8} {'測試R2':<8} {'訓練RMSE':<9} {'驗證RMSE':<9} {'測試RMSE':<9} {'CV分數':<8}"
        )
        print("-" * 100)

        for _, row in comparison_df.iterrows():
            print(
                f"{row['Model']:<15} {row['Train_R2']:<8.4f} {row['Val_R2']:<8.4f} {row['Test_R2']:<8.4f} {row['Train_RMSE']:<9.4f} {row['Val_RMSE']:<9.4f} {row['Test_RMSE']:<9.4f} {row['CV_Mean']:<8.4f}"
            )

        # 找出最佳模型
        best_model = comparison_df.iloc[0]
        print(f"\n最佳模型: {best_model['Model']}")
        print(f"測試集 R2: {best_model['Test_R2']:.4f}")
        print(f"測試集 RMSE: {best_model['Test_RMSE']:.4f}")

        return comparison_df

    def analyze_feature_importance(self):
        """分析特徵重要性"""
        print("\n=== 特徵重要性分析 ===")

        # 計算各模型的特徵重要性
        importance_data = []

        for name, results in self.results.items():
            model = results["model"]

            if hasattr(model, "coef_"):
                # 線性模型的係數
                coef = model.coef_
                for i, (feature, coef_val) in enumerate(zip(self.feature_names, coef)):
                    importance_data.append(
                        {
                            "Model": name,
                            "Feature": feature,
                            "Coefficient": coef_val,
                            "Abs_Coefficient": abs(coef_val),
                        }
                    )

        importance_df = pd.DataFrame(importance_data)

        # 計算平均重要性
        avg_importance = (
            importance_df.groupby("Feature")["Abs_Coefficient"]
            .mean()
            .sort_values(ascending=False)
        )

        print("特徵重要性排名 (按平均絕對係數):")
        print("=" * 60)
        print(f"{'排名':<4} {'特徵':<20} {'平均重要性':<10}")
        print("-" * 60)

        for i, (feature, importance) in enumerate(avg_importance.items(), 1):
            print(f"{i:<4} {feature:<20} {importance:<10.4f}")

        # 儲存特徵重要性
        import os

        os.makedirs("results_no_outliers", exist_ok=True)
        importance_df.to_csv("results_no_outliers/feature_importance.csv", index=False)
        avg_importance.to_csv("results_no_outliers/avg_feature_importance.csv")

        return importance_df, avg_importance

    def visualize_results(self, comparison_df, importance_df, avg_importance):
        """視覺化結果"""
        print("\n=== 生成視覺化圖表 ===")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 模型性能比較
        models = comparison_df["Model"]
        test_r2 = comparison_df["Test_R2"]
        test_rmse = comparison_df["Test_RMSE"]

        x = np.arange(len(models))
        width = 0.35

        axes[0, 0].bar(
            x - width / 2, test_r2, width, label="R2", alpha=0.8, color="skyblue"
        )
        axes[0, 0].set_xlabel("模型")
        axes[0, 0].set_ylabel("R2")
        axes[0, 0].set_title("模型 R2 比較")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 在柱狀圖上顯示數值
        for i, (bar, r2) in enumerate(zip(axes[0, 0].patches, test_r2)):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{r2:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        axes[0, 1].bar(
            x - width / 2, test_rmse, width, label="RMSE", alpha=0.8, color="lightcoral"
        )
        axes[0, 1].set_xlabel("模型")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].set_title("模型 RMSE 比較")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 在柱狀圖上顯示數值
        for i, (bar, rmse) in enumerate(zip(axes[0, 1].patches, test_rmse)):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{rmse:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # 2. 特徵重要性
        top_features = avg_importance.head(10)
        axes[1, 0].barh(
            range(len(top_features)), top_features.values, alpha=0.8, color="lightgreen"
        )
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features.index)
        axes[1, 0].set_xlabel("平均重要性")
        axes[1, 0].set_title("前10個重要特徵")
        axes[1, 0].grid(True, alpha=0.3)

        # 3. 預測 vs 實際值散點圖 (最佳模型)
        best_model_name = comparison_df.iloc[0]["Model"]
        best_results = self.results[best_model_name]
        y_test_pred = best_results["y_test_pred"]

        axes[1, 1].scatter(self.y_test, y_test_pred, alpha=0.6, color="blue")
        axes[1, 1].plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--",
            lw=2,
        )
        axes[1, 1].set_xlabel("實際值")
        axes[1, 1].set_ylabel("預測值")
        axes[1, 1].set_title(f"{best_model_name} 預測 vs 實際值")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        import os

        os.makedirs("results_no_outliers", exist_ok=True)
        plt.savefig(
            "results_no_outliers/baseline_modeling_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print("視覺化圖表已儲存至 results_no_outliers/baseline_modeling_results.png")

    def save_models(self):
        """儲存模型"""
        print("\n=== 儲存模型 ===")

        import os

        os.makedirs("results_no_outliers/models", exist_ok=True)

        for name, results in self.results.items():
            model = results["model"]
            model_path = f"results_no_outliers/models/{name.lower()}_model.pkl"
            joblib.dump(model, model_path)
            print(f"已儲存 {name} 模型至 {model_path}")

    def save_results_csv(self, comparison_df):
        """儲存結果CSV檔案"""
        print("\n=== 儲存結果CSV檔案 ===")

        import os

        os.makedirs("results_no_outliers", exist_ok=True)
        comparison_df.to_csv(
            "results_no_outliers/baseline_modeling_results.csv", index=False
        )
        print("已儲存模型結果到 results_no_outliers/baseline_modeling_results.csv")

    def generate_report(self, comparison_df, importance_df, avg_importance):
        """生成建模報告"""
        print("\n=== 生成建模報告 ===")

        best_model = comparison_df.iloc[0]

        report = f"""
# WineQT 多元線性迴歸建模報告 (保留異常值版本)

## 執行時間
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 資料集資訊
- 訓練集: {self.X_train.shape[0]} 筆 x {self.X_train.shape[1]} 特徵
- 驗證集: {self.X_val.shape[0]} 筆 x {self.X_val.shape[1]} 特徵
- 測試集: {self.X_test.shape[0]} 筆 x {self.X_test.shape[1]} 特徵
- 特徵數量: {len(self.feature_names)} 個 (原始特徵，未新增)

## 模型性能排名

| 排名 | 模型 | 訓練R2 | 驗證R2 | 測試R2 | 訓練RMSE | 驗證RMSE | 測試RMSE | CV分數 |
|------|------|--------|--------|--------|----------|----------|----------|--------|
"""

        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            report += f"| {i} | {row['Model']} | {row['Train_R2']:.4f} | {row['Val_R2']:.4f} | {row['Test_R2']:.4f} | {row['Train_RMSE']:.4f} | {row['Val_RMSE']:.4f} | {row['Test_RMSE']:.4f} | {row['CV_Mean']:.4f} |\n"

        report += f"""
## 最佳模型: {best_model['Model']}

### 性能指標
- **測試集 R2**: {best_model['Test_R2']:.4f}
- **測試集 RMSE**: {best_model['Test_RMSE']:.4f}
- **交叉驗證 R2**: {best_model['CV_Mean']:.4f}

## 特徵重要性 (前10名)

| 排名 | 特徵 | 平均重要性 |
|------|------|------------|
"""

        for i, (feature, importance) in enumerate(avg_importance.head(10).items(), 1):
            report += f"| {i} | {feature} | {importance:.4f} |\n"

        report += f"""
## 結論

### 主要發現
1. **最佳模型**: {best_model['Model']} 在測試集上表現最佳
2. **模型性能**: 所有模型的 R2 都在合理範圍內
3. **特徵重要性**: 識別出最重要的特徵
4. **保留異常值**: 使用完整資料集進行建模

### 建議
1. **模型選擇**: 使用 {best_model['Model']} 作為最終模型
2. **特徵工程**: 基於特徵重要性結果進行進一步優化
3. **模型驗證**: 在驗證集上持續監控模型性能
4. **異常值分析**: 分析保留的異常值對模型性能的影響
"""

        import os

        os.makedirs("results_no_outliers", exist_ok=True)
        with open(
            "results_no_outliers/baseline_modeling_report.md", "w", encoding="utf-8"
        ) as f:
            f.write(report)

        print("建模報告已儲存至 results_no_outliers/baseline_modeling_report.md")
        return report


def main():
    """主執行函數"""
    print("WineQT 多元線性迴歸建模開始 (保留異常值版本)...")

    # 初始化建模器
    modeler = WineModelingNoOutliers()

    # 載入資料
    modeler.load_data()

    # 建立模型
    modeler.build_models()

    # 訓練模型
    modeler.train_models()

    # 比較模型
    comparison_df = modeler.compare_models()

    # 分析特徵重要性
    importance_df, avg_importance = modeler.analyze_feature_importance()

    # 視覺化結果
    modeler.visualize_results(comparison_df, importance_df, avg_importance)

    # 儲存模型
    modeler.save_models()

    # 儲存結果CSV檔案
    modeler.save_results_csv(comparison_df)

    # 生成報告
    modeler.generate_report(comparison_df, importance_df, avg_importance)

    print("\n多元線性迴歸建模完成 (保留異常值版本)！")
    return modeler, comparison_df, importance_df, avg_importance


if __name__ == "__main__":
    modeler, comparison_df, importance_df, avg_importance = main()
