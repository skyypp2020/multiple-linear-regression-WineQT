#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 RFE (Recursive Feature Elimination) 進行特徵選擇分析
列出不同特徵數量下的 RMSE 和 R² 結果
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class RFEAnalyzer:
    """RFE 特徵選擇分析器"""

    def __init__(self):
        """初始化分析器"""
        self.results = {}
        self.feature_names = None

    def load_data(self):
        """載入資料"""
        print("=== 載入資料 ===")

        # 載入訓練資料
        self.X_train = pd.read_csv("processed_data/X_train.csv")
        self.y_train = pd.read_csv("processed_data/y_train.csv").squeeze()

        # 載入驗證資料
        self.X_val = pd.read_csv("processed_data/X_val.csv")
        self.y_val = pd.read_csv("processed_data/y_val.csv").squeeze()

        # 載入測試資料
        self.X_test = pd.read_csv("processed_data/X_test.csv")
        self.y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

        self.feature_names = self.X_train.columns.tolist()

        print(f"訓練集: {self.X_train.shape[0]} 筆 x {self.X_train.shape[1]} 特徵")
        print(f"驗證集: {self.X_val.shape[0]} 筆 x {self.X_val.shape[1]} 特徵")
        print(f"測試集: {self.X_test.shape[0]} 筆 x {self.X_test.shape[1]} 特徵")

        return (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )

    def perform_rfe_analysis(self):
        """執行 RFE 分析"""
        print("\n=== 執行 RFE 特徵選擇分析 ===")

        # 定義模型
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        }

        # 特徵數量範圍 (從 1 到全部特徵)
        n_features_range = range(1, len(self.feature_names) + 1)

        for model_name, model in models.items():
            print(f"\n{model_name} 模型 RFE 分析:")

            model_results = {
                "n_features": [],
                "selected_features": [],
                "train_r2": [],
                "val_r2": [],
                "test_r2": [],
                "train_rmse": [],
                "val_rmse": [],
                "test_rmse": [],
                "cv_scores": [],
            }

            for n_features in n_features_range:
                print(f"  特徵數量: {n_features}")

                # 執行 RFE
                rfe = RFE(estimator=model, n_features_to_select=n_features)
                rfe.fit(self.X_train, self.y_train)

                # 取得選中的特徵
                selected_features = np.array(self.feature_names)[rfe.support_]
                X_train_selected = self.X_train[selected_features]
                X_val_selected = self.X_val[selected_features]
                X_test_selected = self.X_test[selected_features]

                # 訓練模型
                model.fit(X_train_selected, self.y_train)

                # 預測
                y_train_pred = model.predict(X_train_selected)
                y_val_pred = model.predict(X_val_selected)
                y_test_pred = model.predict(X_test_selected)

                # 計算 R²
                train_r2 = r2_score(self.y_train, y_train_pred)
                val_r2 = r2_score(self.y_val, y_val_pred)
                test_r2 = r2_score(self.y_test, y_test_pred)

                # 計算 RMSE
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
                val_rmse = np.sqrt(mean_squared_error(self.y_val, y_val_pred))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

                # 交叉驗證
                cv_scores = cross_val_score(
                    model, X_train_selected, self.y_train, cv=5, scoring="r2"
                )
                cv_mean = cv_scores.mean()

                # 儲存結果
                model_results["n_features"].append(n_features)
                model_results["selected_features"].append(selected_features.tolist())
                model_results["train_r2"].append(train_r2)
                model_results["val_r2"].append(val_r2)
                model_results["test_r2"].append(test_r2)
                model_results["train_rmse"].append(train_rmse)
                model_results["val_rmse"].append(val_rmse)
                model_results["test_rmse"].append(test_rmse)
                model_results["cv_scores"].append(cv_mean)

                print(f"    驗證集 R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}")

            self.results[model_name] = model_results

        return self.results

    def find_best_features(self):
        """找出最佳特徵數量"""
        print("\n=== 找出最佳特徵數量 ===")

        best_results = {}

        for model_name, results in self.results.items():
            # 以驗證集 R² 為標準找出最佳特徵數量
            best_idx = np.argmax(results["val_r2"])
            best_n_features = results["n_features"][best_idx]
            best_val_r2 = results["val_r2"][best_idx]
            best_val_rmse = results["val_rmse"][best_idx]

            best_results[model_name] = {
                "best_n_features": best_n_features,
                "best_val_r2": best_val_r2,
                "best_val_rmse": best_val_rmse,
                "best_features": results["selected_features"][best_idx],
            }

            print(f"{model_name}:")
            print(f"  最佳特徵數量: {best_n_features}")
            print(f"  最佳驗證集 R2: {best_val_r2:.4f}")
            print(f"  最佳驗證集 RMSE: {best_val_rmse:.4f}")
            print(f"  最佳特徵: {results['selected_features'][best_idx]}")

        return best_results

    def create_results_table(self):
        """創建結果表格"""
        print("\n=== 創建 RFE 結果表格 ===")

        # 創建所有結果的 DataFrame
        all_results = []

        for model_name, results in self.results.items():
            for i in range(len(results["n_features"])):
                all_results.append(
                    {
                        "Model": model_name,
                        "N_Features": results["n_features"][i],
                        "Train_R2": results["train_r2"][i],
                        "Val_R2": results["val_r2"][i],
                        "Test_R2": results["test_r2"][i],
                        "Train_RMSE": results["train_rmse"][i],
                        "Val_RMSE": results["val_rmse"][i],
                        "Test_RMSE": results["test_rmse"][i],
                        "CV_Score": results["cv_scores"][i],
                    }
                )

        results_df = pd.DataFrame(all_results)

        # 按驗證集 R2 排序
        results_df = results_df.sort_values(
            ["Model", "Val_R2"], ascending=[True, False]
        )

        print("RFE 結果表格 (按模型和驗證集 R2 排序):")
        print("=" * 120)
        print(
            f"{'模型':<15} {'特徵數':<6} {'訓練R2':<8} {'驗證R2':<8} {'測試R2':<8} {'訓練RMSE':<9} {'驗證RMSE':<9} {'測試RMSE':<9} {'CV分數':<8}"
        )
        print("-" * 120)

        for _, row in results_df.iterrows():
            print(
                f"{row['Model']:<15} {row['N_Features']:<6} {row['Train_R2']:<8.4f} {row['Val_R2']:<8.4f} {row['Test_R2']:<8.4f} {row['Train_RMSE']:<9.4f} {row['Val_RMSE']:<9.4f} {row['Test_RMSE']:<9.4f} {row['CV_Score']:<8.4f}"
            )

        return results_df

    def visualize_results(self, results_df):
        """視覺化結果"""
        print("\n=== 生成 RFE 視覺化圖表 ===")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 驗證集 R² 隨特徵數量變化
        for model_name in self.results.keys():
            model_data = results_df[results_df["Model"] == model_name]
            axes[0, 0].plot(
                model_data["N_Features"],
                model_data["Val_R2"],
                marker="o",
                label=model_name,
                linewidth=2,
            )

        axes[0, 0].set_xlabel("特徵數量")
        axes[0, 0].set_ylabel("驗證集 R2")
        axes[0, 0].set_title("RFE: 驗證集 R2 隨特徵數量變化")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 驗證集 RMSE 隨特徵數量變化
        for model_name in self.results.keys():
            model_data = results_df[results_df["Model"] == model_name]
            axes[0, 1].plot(
                model_data["N_Features"],
                model_data["Val_RMSE"],
                marker="s",
                label=model_name,
                linewidth=2,
            )

        axes[0, 1].set_xlabel("特徵數量")
        axes[0, 1].set_ylabel("驗證集 RMSE")
        axes[0, 1].set_title("RFE: 驗證集 RMSE 隨特徵數量變化")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 最佳特徵數量的 R² 比較
        best_features = results_df.groupby("Model")["Val_R2"].idxmax()
        best_data = results_df.loc[best_features]

        models = best_data["Model"]
        val_r2 = best_data["Val_R2"]
        n_features = best_data["N_Features"]

        x = np.arange(len(models))
        bars = axes[1, 0].bar(
            x,
            val_r2,
            alpha=0.7,
            color=["skyblue", "lightgreen", "lightcoral", "orange"],
        )
        axes[1, 0].set_xlabel("模型")
        axes[1, 0].set_ylabel("最佳驗證集 R2")
        axes[1, 0].set_title("RFE: 各模型最佳驗證集 R2")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 在柱狀圖上顯示特徵數量和R2值
        for i, (bar, r2, n_feat) in enumerate(zip(bars, val_r2, n_features)):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{n_feat}特徵\nR2={r2:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 4. 最佳特徵數量的 RMSE 比較
        val_rmse = best_data["Val_RMSE"]

        bars = axes[1, 1].bar(
            x,
            val_rmse,
            alpha=0.7,
            color=["skyblue", "lightgreen", "lightcoral", "orange"],
        )
        axes[1, 1].set_xlabel("模型")
        axes[1, 1].set_ylabel("最佳驗證集 RMSE")
        axes[1, 1].set_title("RFE: 各模型最佳驗證集 RMSE")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # 在柱狀圖上顯示特徵數量和RMSE值
        for i, (bar, rmse, n_feat) in enumerate(zip(bars, val_rmse, n_features)):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{n_feat}特徵\nRMSE={rmse:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig("results/plots/rfe_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("RFE 分析圖表已儲存至 results/plots/rfe_analysis.png")

    def generate_rfe_report(self, results_df, best_results):
        """生成 RFE 報告"""
        print("\n=== 生成 RFE 報告 ===")

        # 找出整體最佳模型
        best_overall_idx = results_df["Val_R2"].idxmax()
        best_overall = results_df.loc[best_overall_idx]

        report = f"""
# WineQT RFE 特徵選擇分析報告

## 執行時間
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 整體最佳模型
- **模型**: {best_overall['Model']}
- **特徵數量**: {best_overall['N_Features']}
- **驗證集 R2**: {best_overall['Val_R2']:.4f}
- **驗證集 RMSE**: {best_overall['Val_RMSE']:.4f}
- **測試集 R2**: {best_overall['Test_R2']:.4f}
- **測試集 RMSE**: {best_overall['Test_RMSE']:.4f}

## 各模型最佳結果

| 模型 | 最佳特徵數 | 驗證集R2 | 驗證集RMSE | 測試集R2 | 測試集RMSE |
|------|------------|----------|------------|----------|------------|
"""

        for model_name, best in best_results.items():
            model_data = results_df[results_df["Model"] == model_name]
            best_idx = model_data["Val_R2"].idxmax()
            best_row = model_data.loc[best_idx]

            report += f"| {model_name} | {best_row['N_Features']} | {best_row['Val_R2']:.4f} | {best_row['Val_RMSE']:.4f} | {best_row['Test_R2']:.4f} | {best_row['Test_RMSE']:.4f} |\n"

        report += f"""
## 特徵選擇效果分析

### 特徵數量對性能的影響
"""

        for model_name in self.results.keys():
            model_data = results_df[results_df["Model"] == model_name]
            min_features = model_data["N_Features"].min()
            max_features = model_data["N_Features"].max()
            best_r2 = model_data["Val_R2"].max()
            worst_r2 = model_data["Val_R2"].min()

            report += f"""
**{model_name}**:
- 特徵數量範圍: {min_features} - {max_features}
- 最佳 R2: {best_r2:.4f}
- 最差 R2: {worst_r2:.4f}
- R2 改善: {best_r2 - worst_r2:.4f}
"""

        report += f"""
## 結論

### 主要發現
1. **最佳模型**: {best_overall['Model']} 在 {best_overall['N_Features']} 個特徵時表現最佳
2. **特徵選擇效果**: RFE 有效識別出最重要的特徵組合
3. **模型比較**: 不同模型對特徵數量的敏感度不同
4. **性能提升**: 特徵選擇可以提升模型性能

### 建議
1. **模型選擇**: 使用 {best_overall['Model']} 作為最終模型
2. **特徵數量**: 使用 {best_overall['N_Features']} 個特徵
3. **特徵工程**: 基於 RFE 結果進一步優化特徵
4. **模型驗證**: 在測試集上驗證最終模型性能
"""

        with open("results/rfe_analysis_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("RFE 分析報告已儲存至 results/rfe_analysis_report.md")
        return report


def main():
    """主執行函數"""
    print("WineQT RFE 特徵選擇分析開始...")

    # 初始化分析器
    analyzer = RFEAnalyzer()

    # 載入資料
    analyzer.load_data()

    # 執行 RFE 分析
    analyzer.perform_rfe_analysis()

    # 找出最佳特徵數量
    best_results = analyzer.find_best_features()

    # 創建結果表格
    results_df = analyzer.create_results_table()

    # 視覺化結果
    analyzer.visualize_results(results_df)

    # 生成報告
    analyzer.generate_rfe_report(results_df, best_results)

    # 保存結果CSV檔案
    results_df.to_csv("rfe_results.csv", index=False)
    print("RFE 結果已保存到 rfe_results.csv")

    print("\nRFE 特徵選擇分析完成！")
    return analyzer, results_df, best_results


if __name__ == "__main__":
    analyzer, results_df, best_results = main()
