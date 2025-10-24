#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit應用程式 - 葡萄酒品質預測模型分析
展示模型比隨機亂猜更準確的證明及相關分析結果
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# 設定頁面配置
st.set_page_config(
    page_title="葡萄酒品質預測模型分析",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class StreamlitApp:
    """Streamlit應用程式主類"""

    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """載入資料"""
        try:
            # 載入原始資料
            self.data = pd.read_csv("datasets/WineQT.csv")

            # 載入處理後的資料
            self.processed_data = {
                "with_outliers": pd.read_csv(
                    "processed_data_no_outliers/wine_processed.csv"
                ),
                "without_outliers": pd.read_csv("processed_data/wine_processed.csv"),
            }

            # 載入測試資料
            self.X_test = pd.read_csv("processed_data_no_outliers/X_test.csv")
            self.y_test = pd.read_csv("processed_data_no_outliers/y_test.csv").squeeze()

            return True
        except FileNotFoundError as e:
            st.error(f"找不到資料檔案: {e}")
            return False

    def load_models(self):
        """載入模型"""
        try:
            # 載入保留異常值版本的模型
            model_dir = "results_no_outliers/models"
            if os.path.exists(model_dir):
                self.models["ridge"] = joblib.load(f"{model_dir}/ridge_model.pkl")
                self.models["linear"] = joblib.load(
                    f"{model_dir}/linearregression_model.pkl"
                )
                self.models["lasso"] = joblib.load(f"{model_dir}/lasso_model.pkl")
                self.models["elastic"] = joblib.load(
                    f"{model_dir}/elasticnet_model.pkl"
                )
                return True
            else:
                st.error("找不到模型檔案")
                return False
        except Exception as e:
            st.error(f"載入模型時發生錯誤: {e}")
            return False

    def show_data_preprocessing(self):
        """顯示資料預處理資訊"""
        st.header("📊 資料預處理結果")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("原始資料統計")
            st.dataframe(self.data.describe(), use_container_width=True)

            st.subheader("資料品質檢查")
            quality_info = {
                "總樣本數": len(self.data),
                "特徵數": len(self.data.columns) - 1,
                "缺失值": self.data.isnull().sum().sum(),
                "重複值": self.data.duplicated().sum(),
                "品質分數範圍": f"{self.data['quality'].min()} - {self.data['quality'].max()}",
            }

            for key, value in quality_info.items():
                st.metric(key, value)

        with col2:
            st.subheader("資料分布")

            # 品質分數分布
            fig = px.histogram(
                self.data,
                x="quality",
                title="品質分數分布",
                color_discrete_sequence=["#1f77b4"],
            )
            fig.update_layout(
                xaxis_title="品質分數", yaxis_title="頻率", showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # 預處理比較
        st.subheader("預處理版本比較")

        comparison_data = {
            "版本": ["保留異常值", "移除異常值"],
            "樣本數": [
                len(self.processed_data["with_outliers"]),
                len(self.processed_data["without_outliers"]),
            ],
            "特徵數": [
                len(self.processed_data["with_outliers"].columns) - 1,
                len(self.processed_data["without_outliers"].columns) - 1,
            ],
            "品質分數範圍": [
                f"{self.processed_data['with_outliers']['quality'].min()}-{self.processed_data['with_outliers']['quality'].max()}",
                f"{self.processed_data['without_outliers']['quality'].min()}-{self.processed_data['without_outliers']['quality'].max()}",
            ],
        }

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    def show_model_performance(self):
        """顯示模型效能"""
        st.header("🎯 模型效能分析")

        # 載入模型結果
        try:
            # 載入保留異常值版本的結果
            results_file = "results_no_outliers/baseline_modeling_results.csv"
            if os.path.exists(results_file):
                results_df = pd.read_csv(results_file)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("模型效能比較")
                    st.dataframe(results_df, use_container_width=True)

                with col2:
                    st.subheader("最佳模型")
                    best_model = results_df.loc[results_df["Test_R2"].idxmax()]
                    st.metric("最佳模型", best_model["Model"])
                    st.metric("R2", f"{best_model['Test_R2']:.4f}")
                    st.metric("RMSE", f"{best_model['Test_RMSE']:.4f}")
                    st.metric("MAE", f"{best_model['Test_MAE']:.4f}")

                # 效能視覺化
                st.subheader("效能視覺化")

                fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=("R2 比較", "RMSE 比較", "MAE 比較"),
                    specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
                )

                models = results_df["Model"]
                r2_values = results_df["Test_R2"]
                rmse_values = results_df["Test_RMSE"]
                mae_values = results_df["Test_MAE"]

                fig.add_trace(
                    go.Bar(x=models, y=r2_values, name="R2", marker_color="blue"),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Bar(x=models, y=rmse_values, name="RMSE", marker_color="red"),
                    row=1,
                    col=2,
                )

                fig.add_trace(
                    go.Bar(x=models, y=mae_values, name="MAE", marker_color="green"),
                    row=1,
                    col=3,
                )

                fig.update_layout(
                    height=400, showlegend=False, title_text="模型效能比較"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("找不到模型結果檔案")

        except Exception as e:
            st.error(f"載入模型結果時發生錯誤: {e}")

    def show_model_vs_random(self):
        """顯示模型與隨機預測比較"""
        st.header("🎲 模型 vs 隨機預測比較")

        if not self.load_models():
            st.error("無法載入模型")
            return

        # 生成隨機預測
        np.random.seed(42)
        random_predictions = np.random.uniform(3, 8, size=len(self.y_test))

        # 模型預測
        model_predictions = self.models["ridge"].predict(self.X_test)

        # 計算指標
        def calculate_metrics(y_true, y_pred, method_name):
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            # 分類準確率
            y_true_rounded = np.round(y_true).astype(int)
            y_pred_rounded = np.round(y_pred).astype(int)
            y_true_clipped = np.clip(y_true_rounded, 3, 8)
            y_pred_clipped = np.clip(y_pred_rounded, 3, 8)

            exact_accuracy = accuracy_score(y_true_clipped, y_pred_clipped)
            diff = np.abs(y_true_clipped - y_pred_clipped)
            within_one_accuracy = np.mean(diff <= 1)

            return {
                "method": method_name,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "exact_accuracy": exact_accuracy,
                "within_one_accuracy": within_one_accuracy,
            }

        # 計算比較結果
        model_metrics = calculate_metrics(self.y_test, model_predictions, "Ridge模型")
        random_metrics = calculate_metrics(self.y_test, random_predictions, "隨機預測")

        # 顯示比較表格
        st.subheader("詳細比較結果")
        comparison_df = pd.DataFrame([model_metrics, random_metrics])
        st.dataframe(comparison_df, use_container_width=True)

        # 關鍵指標改善
        st.subheader("關鍵指標改善")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            r2_improvement = model_metrics["r2"] - random_metrics["r2"]
            st.metric("R2 改善", f"{r2_improvement:.4f}", delta=f"{r2_improvement:.4f}")

        with col2:
            accuracy_improvement = (
                model_metrics["exact_accuracy"] - random_metrics["exact_accuracy"]
            )
            st.metric(
                "完全正確預測率改善",
                f"{accuracy_improvement:.2%}",
                delta=f"{accuracy_improvement:.2%}",
            )

        with col3:
            within_one_improvement = (
                model_metrics["within_one_accuracy"]
                - random_metrics["within_one_accuracy"]
            )
            st.metric(
                "誤差±1範圍內改善",
                f"{within_one_improvement:.2%}",
                delta=f"{within_one_improvement:.2%}",
            )

        with col4:
            error_reduction = (
                (random_metrics["mae"] - model_metrics["mae"])
                / random_metrics["mae"]
                * 100
            )
            st.metric(
                "平均誤差減少",
                f"{error_reduction:.1f}%",
                delta=f"{error_reduction:.1f}%",
            )

        # 視覺化比較
        st.subheader("視覺化比較")

        # 1. 預測值 vs 實際值散點圖
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Ridge模型預測",
                "隨機預測",
                "誤差分布比較",
                "各品質等級準確率",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}],
            ],
        )

        # Ridge模型散點圖
        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=model_predictions,
                mode="markers",
                name="Ridge模型",
                marker=dict(color="blue", opacity=0.6),
            ),
            row=1,
            col=1,
        )

        # 隨機預測散點圖
        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=random_predictions,
                mode="markers",
                name="隨機預測",
                marker=dict(color="red", opacity=0.6),
            ),
            row=1,
            col=2,
        )

        # 誤差分布比較
        model_errors = np.abs(self.y_test - model_predictions)
        random_errors = np.abs(self.y_test - random_predictions)

        fig.add_trace(
            go.Histogram(
                x=model_errors, name="Ridge模型誤差", marker_color="blue", opacity=0.7
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=random_errors, name="隨機預測誤差", marker_color="red", opacity=0.7
            ),
            row=2,
            col=1,
        )

        # 各品質等級準確率
        quality_levels = sorted(self.y_test.unique())
        model_accuracies = []
        random_accuracies = []

        for level in quality_levels:
            mask = self.y_test == level
            if np.sum(mask) > 0:
                model_pred_rounded = np.round(model_predictions[mask]).astype(int)
                model_acc = accuracy_score(
                    self.y_test[mask].astype(int), model_pred_rounded
                )
                model_accuracies.append(model_acc)

                random_pred_rounded = np.round(random_predictions[mask]).astype(int)
                random_acc = accuracy_score(
                    self.y_test[mask].astype(int), random_pred_rounded
                )
                random_accuracies.append(random_acc)
            else:
                model_accuracies.append(0)
                random_accuracies.append(0)

        fig.add_trace(
            go.Bar(
                x=quality_levels,
                y=model_accuracies,
                name="Ridge模型",
                marker_color="blue",
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=quality_levels,
                y=random_accuracies,
                name="隨機預測",
                marker_color="red",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800, title_text="模型與隨機預測詳細比較", showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # 統計顯著性分析
        st.subheader("統計顯著性分析")

        better_predictions = np.sum(model_errors < random_errors)
        total_samples = len(model_errors)
        better_ratio = better_predictions / total_samples

        improvement = (
            (np.mean(random_errors) - np.mean(model_errors))
            / np.mean(random_errors)
            * 100
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "模型更準確的樣本",
                f"{better_predictions}/{total_samples}",
                f"{better_ratio:.1%}",
            )

        with col2:
            st.metric("整體改善幅度", f"{improvement:.1f}%")

        with col3:
            st.metric(
                "平均誤差減少", f"{np.mean(random_errors) - np.mean(model_errors):.3f}"
            )

    def show_rfe_analysis(self):
        """顯示RFE分析結果"""
        st.header("🔍 RFE (遞歸特徵消除) 分析")

        try:
            # 載入RFE結果
            rfe_file = "rfe_results.csv"
            if os.path.exists(rfe_file):
                rfe_df = pd.read_csv(rfe_file)

                st.subheader("RFE結果表格")
                st.dataframe(rfe_df, use_container_width=True)

                # RFE視覺化
                st.subheader("RFE分析視覺化")

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("R2 vs 特徵數量", "RMSE vs 特徵數量"),
                    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
                )

                for model in rfe_df["Model"].unique():
                    model_data = rfe_df[rfe_df["Model"] == model]

                    fig.add_trace(
                        go.Scatter(
                            x=model_data["N_Features"],
                            y=model_data["Val_R2"],
                            mode="lines+markers",
                            name=f"{model} R2",
                            line=dict(width=2),
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=model_data["N_Features"],
                            y=model_data["Val_RMSE"],
                            mode="lines+markers",
                            name=f"{model} RMSE",
                            line=dict(width=2),
                        ),
                        row=1,
                        col=2,
                    )

                fig.update_layout(height=500, title_text="RFE分析結果")

                st.plotly_chart(fig, use_container_width=True)

                # 最佳特徵數量
                st.subheader("最佳特徵數量")
                best_features = rfe_df.loc[rfe_df["Val_R2"].idxmax()]
                st.metric("最佳特徵數量", best_features["N_Features"])
                st.metric("最佳R2", f"{best_features['Val_R2']:.4f}")
                st.metric("對應RMSE", f"{best_features['Val_RMSE']:.4f}")

            else:
                st.warning("找不到RFE結果檔案")

        except Exception as e:
            st.error(f"載入RFE結果時發生錯誤: {e}")

    def show_feature_importance(self):
        """顯示特徵重要性"""
        st.header("📈 特徵重要性分析")

        try:
            # 載入特徵重要性
            importance_file = "results/data/all_features_importance.csv"
            if os.path.exists(importance_file):
                importance_df = pd.read_csv(importance_file)

                st.subheader("特徵重要性表格")
                st.dataframe(importance_df, use_container_width=True)

                # 特徵重要性視覺化
                st.subheader("特徵重要性視覺化")

                # 平均重要性
                avg_importance = importance_df.set_index("Feature")[
                    "Average_Importance"
                ].sort_values(ascending=True)

                fig = px.bar(
                    x=avg_importance.values,
                    y=avg_importance.index,
                    orientation="h",
                    title="平均特徵重要性",
                    color=avg_importance.values,
                    color_continuous_scale="viridis",
                )

                fig.update_layout(xaxis_title="重要性", yaxis_title="特徵", height=500)

                st.plotly_chart(fig, use_container_width=True)

                # 各模型特徵重要性比較
                st.subheader("各模型特徵重要性比較")

                model_columns = [
                    col
                    for col in importance_df.columns
                    if col not in ["Feature", "Average_Importance"]
                ]

                fig = go.Figure()

                for col in model_columns:
                    fig.add_trace(
                        go.Bar(
                            x=importance_df["Feature"],
                            y=importance_df[col],
                            name=col,
                            opacity=0.8,
                        )
                    )

                fig.update_layout(
                    title="各模型特徵重要性比較",
                    xaxis_title="特徵",
                    yaxis_title="重要性",
                    barmode="group",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("找不到特徵重要性檔案")

        except Exception as e:
            st.error(f"載入特徵重要性時發生錯誤: {e}")

    def show_outlier_analysis(self):
        """顯示異常值分析"""
        st.header("🔍 異常值分析")

        try:
            # 載入異常值分析結果
            outlier_file = "processed_data/outliers_removed.csv"
            if os.path.exists(outlier_file):
                outliers_df = pd.read_csv(outlier_file)

                st.subheader("移除的異常值統計")
                st.metric("移除的異常值數量", len(outliers_df))

                # 異常值特徵分析
                st.subheader("異常值特徵分析")

                # 比較異常值與正常值的特徵分布
                normal_data = self.processed_data["without_outliers"]

                comparison_features = [
                    "alcohol",
                    "volatile acidity",
                    "citric acid",
                    "residual sugar",
                ]

                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=comparison_features,
                    specs=[
                        [{"type": "histogram"}, {"type": "histogram"}],
                        [{"type": "histogram"}, {"type": "histogram"}],
                    ],
                )

                for i, feature in enumerate(comparison_features):
                    row = i // 2 + 1
                    col = i % 2 + 1

                    fig.add_trace(
                        go.Histogram(
                            x=normal_data[feature],
                            name="正常值",
                            marker_color="blue",
                            opacity=0.7,
                        ),
                        row=row,
                        col=col,
                    )

                    fig.add_trace(
                        go.Histogram(
                            x=outliers_df[feature],
                            name="異常值",
                            marker_color="red",
                            opacity=0.7,
                        ),
                        row=row,
                        col=col,
                    )

                fig.update_layout(
                    height=600, title_text="異常值與正常值特徵分布比較", showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # 異常值統計表格
                st.subheader("異常值統計")
                outlier_stats = outliers_df.describe()
                st.dataframe(outlier_stats, use_container_width=True)

            else:
                st.warning("找不到異常值分析檔案")

        except Exception as e:
            st.error(f"載入異常值分析時發生錯誤: {e}")

    def run(self):
        """執行Streamlit應用程式"""
        st.title("🍷 葡萄酒品質預測模型分析")
        st.markdown("---")

        # 載入資料
        if not self.load_data():
            st.error("無法載入資料，請檢查檔案路徑")
            return

        # 側邊欄導航
        st.sidebar.title("📋 分析選單")

        analysis_options = {
            "資料預處理": self.show_data_preprocessing,
            "模型效能": self.show_model_performance,
            "模型 vs 隨機預測": self.show_model_vs_random,
            "RFE分析": self.show_rfe_analysis,
            "特徵重要性": self.show_feature_importance,
            "異常值分析": self.show_outlier_analysis,
        }

        selected_analysis = st.sidebar.selectbox(
            "選擇分析項目", list(analysis_options.keys())
        )

        # 執行選定的分析
        analysis_options[selected_analysis]()

        # 頁腳
        st.markdown("---")
        st.markdown(
            "**分析完成時間:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        st.markdown("**資料來源:** WineQT 資料集")
        st.markdown("**分析方法:** CRISP-DM 框架")


def main():
    """主執行函數"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
