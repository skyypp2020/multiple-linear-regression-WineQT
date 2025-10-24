#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析所有特徵的重要性數值
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def analyze_all_feature_importance():
    """分析所有特徵的重要性"""
    print("=== 所有特徵重要性分析 ===")

    # 載入資料
    X_train = pd.read_csv("processed_data/X_train.csv")

    # 載入所有模型
    models = {
        "LinearRegression": joblib.load("results/models/linearregression_model.pkl"),
        "Ridge": joblib.load("results/models/ridge_model.pkl"),
        "Lasso": joblib.load("results/models/lasso_model.pkl"),
        "ElasticNet": joblib.load("results/models/elasticnet_model.pkl"),
    }

    # 創建特徵重要性比較表
    feature_importance_df = pd.DataFrame()
    feature_importance_df["Feature"] = X_train.columns

    print("各模型特徵重要性 (係數值):")
    print("=" * 80)

    for model_name, model in models.items():
        if hasattr(model, "coef_"):
            coefficients = model.coef_
            feature_importance_df[f"{model_name}_Coefficient"] = coefficients
            feature_importance_df[f"{model_name}_Abs_Coefficient"] = np.abs(
                coefficients
            )

            print(f"\n{model_name} 模型:")
            print("-" * 50)

            # 按絕對值排序
            sorted_features = sorted(
                zip(X_train.columns, coefficients),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            for i, (feature, coef) in enumerate(sorted_features, 1):
                print(f"{i:2d}. {feature:25s}: {coef:8.4f}")

    # 計算平均重要性
    abs_cols = [
        col for col in feature_importance_df.columns if "Abs_Coefficient" in col
    ]
    feature_importance_df["Average_Importance"] = feature_importance_df[abs_cols].mean(
        axis=1
    )

    # 按平均重要性排序
    feature_importance_df = feature_importance_df.sort_values(
        "Average_Importance", ascending=False
    )

    print(f"\n所有特徵平均重要性排序:")
    print("=" * 80)
    for idx, row in feature_importance_df.iterrows():
        print(
            f"{row.name+1:2d}. {row['Feature']:25s}: {row['Average_Importance']:8.4f}"
        )

    # 儲存結果
    feature_importance_df.to_csv(
        "results/data/all_features_importance.csv", index=False
    )
    print(f"\n特徵重要性已儲存至 results/data/all_features_importance.csv")

    return feature_importance_df


def visualize_feature_importance(feature_importance_df):
    """視覺化特徵重要性"""
    print("\n=== 生成特徵重要性視覺化圖表 ===")

    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 平均重要性條形圖
    top_features = feature_importance_df.head(15)
    axes[0, 0].barh(range(len(top_features)), top_features["Average_Importance"])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features["Feature"])
    axes[0, 0].set_xlabel("平均重要性")
    axes[0, 0].set_title("特徵平均重要性 (前 15 名)")
    axes[0, 0].invert_yaxis()

    # 2. 各模型係數比較 (前 10 名)
    top_10_features = feature_importance_df.head(10)["Feature"]
    model_names = ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]

    x = np.arange(len(top_10_features))
    width = 0.2

    for i, model_name in enumerate(model_names):
        coef_col = f"{model_name}_Coefficient"
        if coef_col in feature_importance_df.columns:
            model_data = feature_importance_df[
                feature_importance_df["Feature"].isin(top_10_features)
            ]
            model_data = model_data.sort_values("Average_Importance", ascending=True)
            axes[0, 1].barh(
                x + i * width, model_data[coef_col], width, label=model_name, alpha=0.8
            )

    axes[0, 1].set_yticks(x + width * 1.5)
    axes[0, 1].set_yticklabels(top_10_features)
    axes[0, 1].set_xlabel("係數值")
    axes[0, 1].set_title("各模型特徵係數比較 (前 10 名)")
    axes[0, 1].legend()
    axes[0, 1].invert_yaxis()

    # 3. 特徵重要性熱力圖
    coef_matrix = feature_importance_df[
        [
            "LinearRegression_Coefficient",
            "Ridge_Coefficient",
            "Lasso_Coefficient",
            "ElasticNet_Coefficient",
        ]
    ].T
    coef_matrix.columns = feature_importance_df["Feature"]

    sns.heatmap(
        coef_matrix.iloc[:, :15],
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("特徵係數熱力圖 (前 15 名)")
    axes[1, 0].set_xlabel("特徵")
    axes[1, 0].set_ylabel("模型")

    # 4. 正負係數分布
    positive_coef = (feature_importance_df["Average_Importance"] > 0).sum()
    negative_coef = (feature_importance_df["Average_Importance"] < 0).sum()
    zero_coef = (feature_importance_df["Average_Importance"] == 0).sum()

    axes[1, 1].pie(
        [positive_coef, negative_coef, zero_coef],
        labels=["正向影響", "負向影響", "無影響"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1, 1].set_title("特徵影響方向分布")

    plt.tight_layout()
    plt.savefig(
        "results/plots/comprehensive_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(
        "特徵重要性視覺化圖表已儲存至 results/plots/comprehensive_feature_importance.png"
    )


def generate_feature_importance_report(feature_importance_df):
    """生成特徵重要性報告"""
    print("\n=== 生成特徵重要性報告 ===")

    report = f"""
# WineQT 特徵重要性分析報告

## 概述
- 分析時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 特徵總數: {len(feature_importance_df)} 個
- 分析模型: LinearRegression, Ridge, Lasso, ElasticNet

## 特徵重要性排名 (按平均絕對係數值)

| 排名 | 特徵名稱 | 平均重要性 | LinearRegression | Ridge | Lasso | ElasticNet |
|------|----------|------------|------------------|-------|-------|------------|
"""

    for idx, row in feature_importance_df.iterrows():
        report += f"| {idx+1} | {row['Feature']} | {row['Average_Importance']:.4f} | "
        report += f"{row['LinearRegression_Coefficient']:.4f} | "
        report += f"{row['Ridge_Coefficient']:.4f} | "
        report += f"{row['Lasso_Coefficient']:.4f} | "
        report += f"{row['ElasticNet_Coefficient']:.4f} |\n"

    # 分析正向和負向影響的特徵
    positive_features = feature_importance_df[
        feature_importance_df["Average_Importance"] > 0
    ]
    negative_features = feature_importance_df[
        feature_importance_df["Average_Importance"] < 0
    ]

    report += f"""
## 特徵影響分析

### 正向影響特徵 (提升品質)
{len(positive_features)} 個特徵對葡萄酒品質有正向影響:

"""

    for idx, row in positive_features.head(10).iterrows():
        report += f"- {row['Feature']}: {row['Average_Importance']:.4f}\n"

    report += f"""
### 負向影響特徵 (降低品質)
{len(negative_features)} 個特徵對葡萄酒品質有負向影響:

"""

    for idx, row in negative_features.head(10).iterrows():
        report += f"- {row['Feature']}: {row['Average_Importance']:.4f}\n"

    report += """
## 模型一致性分析

### 高一致性特徵 (所有模型都認為重要)
- 這些特徵在所有模型中都有較高的絕對係數值
- 表示這些特徵對預測結果有穩定且重要的影響

### 模型差異特徵
- 某些特徵在不同模型中的重要性差異較大
- 可能表示這些特徵與其他特徵存在共線性問題

## 建議

1. **重點關注前 10 名特徵**: 這些特徵對模型預測最為重要
2. **特徵工程優化**: 可以基於重要性排名進行特徵選擇
3. **模型改進**: 考慮使用特徵重要性進行模型調優
4. **業務解釋**: 結合葡萄酒學知識解釋特徵重要性的合理性

## 結論

特徵重要性分析顯示了不同化學屬性對葡萄酒品質的影響程度，為模型改進和業務理解提供了重要參考。
"""

    with open("results/feature_importance_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("特徵重要性報告已儲存至 results/feature_importance_report.md")
    return report


def main():
    """主執行函數"""
    print("WineQT 特徵重要性分析開始...")

    # 分析特徵重要性
    feature_importance_df = analyze_all_feature_importance()

    # 視覺化
    visualize_feature_importance(feature_importance_df)

    # 生成報告
    generate_feature_importance_report(feature_importance_df)

    print("\n特徵重要性分析完成！")
    return feature_importance_df


if __name__ == "__main__":
    feature_importance_df = main()
