#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WineQT 資料前處理腳本 (保留異常值版本)
不移除異常值，不新增特徵，只進行基本清理
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class WineDataPreprocessorNoOutliers:
    """WineQT 資料前處理類別 (保留異常值版本)"""

    def __init__(self, data_path="datasets/WineQT.csv"):
        """初始化資料前處理器"""
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.scaler = StandardScaler()

    def load_data(self):
        """載入資料集"""
        print("=== 載入資料集 ===")
        self.df = pd.read_csv(self.data_path)
        print(f"資料集形狀: {self.df.shape}")
        print(f"資料集大小: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return self.df

    def explore_data(self):
        """探索性資料分析"""
        print("\n=== 資料探索 ===")

        # 基本資訊
        print("資料集基本資訊:")
        print(f"行數: {self.df.shape[0]}")
        print(f"列數: {self.df.shape[1]}")
        print(f"記憶體使用: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # 資料型別
        print("\n資料型別:")
        print(self.df.dtypes)

        # 缺失值檢查
        print("\n缺失值統計:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame(
            {"缺失數量": missing_data, "缺失百分比": missing_percent}
        )
        print(missing_df[missing_df["缺失數量"] > 0])

        # 重複資料檢查
        print(f"\n重複資料筆數: {self.df.duplicated().sum()}")

        # 數值統計
        print("\n數值統計摘要:")
        print(self.df.describe())

        # 目標變數分布
        print(f"\n目標變數 (quality) 分布:")
        print(self.df["quality"].value_counts().sort_index())

        return self.df

    def clean_data(self):
        """清理資料 (保留異常值)"""
        print("\n=== 資料清理 (保留異常值) ===")
        self.processed_df = self.df.copy()

        # 1. 移除重複資料
        initial_rows = len(self.processed_df)
        self.processed_df = self.processed_df.drop_duplicates()
        removed_duplicates = initial_rows - len(self.processed_df)
        print(f"移除重複資料: {removed_duplicates} 筆")

        # 2. 處理缺失值 (如果有的話)
        missing_before = self.processed_df.isnull().sum().sum()
        if missing_before > 0:
            # 數值型欄位用中位數填補
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.processed_df[col].isnull().sum() > 0:
                    self.processed_df[col].fillna(
                        self.processed_df[col].median(), inplace=True
                    )
            print(f"填補缺失值: {missing_before} 個")

        # 3. 不移除異常值 (保留所有資料)
        print("保留所有資料 (不移除異常值)")

        print(f"清理後資料形狀: {self.processed_df.shape}")
        return self.processed_df

    def prepare_features_target(self):
        """準備特徵和目標變數 (不新增特徵)"""
        print("\n=== 準備特徵和目標變數 (不新增特徵) ===")

        # 移除 ID 欄位
        if "Id" in self.processed_df.columns:
            self.processed_df = self.processed_df.drop("Id", axis=1)
            print("移除 ID 欄位")

        # 分離特徵和目標變數
        X = self.processed_df.drop("quality", axis=1)
        y = self.processed_df["quality"]

        print(f"特徵數量: {X.shape[1]} (原始特徵，未新增)")
        print(f"樣本數量: {X.shape[0]}")
        print(f"目標變數分布:")
        print(y.value_counts().sort_index())

        return X, y

    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """分割資料集"""
        print("\n=== 資料分割 ===")

        # 第一次分割: 訓練+驗證 vs 測試
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 第二次分割: 訓練 vs 驗證
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=y_temp,
        )

        print(f"訓練集: {X_train.shape[0]} 筆 ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"驗證集: {X_val.shape[0]} 筆 ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"測試集: {X_test.shape[0]} 筆 ({X_test.shape[0]/len(X)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train, X_val, X_test):
        """特徵標準化"""
        print("\n=== 特徵標準化 ===")

        # 擬合標準化器
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # 轉換回 DataFrame
        feature_names = X_train.columns
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=feature_names, index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            X_val_scaled, columns=feature_names, index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=feature_names, index=X_test.index
        )

        print("特徵標準化完成")
        print(
            f"標準化後特徵範圍: [{X_train_scaled.min().min():.3f}, {X_train_scaled.max().max():.3f}]"
        )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """儲存處理後的資料"""
        print("\n=== 儲存處理後資料 ===")

        # 創建結果目錄
        import os

        os.makedirs("processed_data_no_outliers", exist_ok=True)

        # 儲存分割後的資料
        X_train.to_csv("processed_data_no_outliers/X_train.csv", index=False)
        X_val.to_csv("processed_data_no_outliers/X_val.csv", index=False)
        X_test.to_csv("processed_data_no_outliers/X_test.csv", index=False)
        y_train.to_csv("processed_data_no_outliers/y_train.csv", index=False)
        y_val.to_csv("processed_data_no_outliers/y_val.csv", index=False)
        y_test.to_csv("processed_data_no_outliers/y_test.csv", index=False)

        # 儲存完整的處理後資料
        self.processed_df.to_csv(
            "processed_data_no_outliers/wine_processed.csv", index=False
        )

        print("資料已儲存至 processed_data_no_outliers/ 目錄")
        print("檔案列表:")
        print("- X_train.csv, X_val.csv, X_test.csv (特徵資料)")
        print("- y_train.csv, y_val.csv, y_test.csv (目標變數)")
        print("- wine_processed.csv (完整處理後資料)")

    def generate_report(self):
        """生成資料前處理報告"""
        print("\n=== 資料前處理報告 ===")

        report = f"""
# WineQT 資料前處理報告 (保留異常值版本)

## 原始資料資訊
- 資料集大小: {self.df.shape[0]} 筆 x {self.df.shape[1]} 欄
- 記憶體使用: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

## 資料清理結果
- 移除重複資料: {self.df.shape[0] - self.processed_df.shape[0]} 筆
- 保留異常值: 不移除任何異常值
- 最終資料大小: {self.processed_df.shape[0]} 筆 x {self.processed_df.shape[1]} 欄

## 特徵工程
- 新增特徵: 0 個 (不新增任何特徵)
- 使用原始特徵: {self.processed_df.shape[1] - 1} 個

## 資料分割
- 訓練集: 70%
- 驗證集: 15%  
- 測試集: 15%

## 資料品質
- 無缺失值
- 已標準化
- 保留所有資料 (包括異常值)
- 已移除重複資料
        """

        with open(
            "processed_data_no_outliers/preprocessing_report.md", "w", encoding="utf-8"
        ) as f:
            f.write(report)

        print("前處理報告已儲存至 processed_data_no_outliers/preprocessing_report.md")
        return report


def main():
    """主執行函數"""
    print("WineQT 資料前處理開始 (保留異常值版本)...")

    # 初始化前處理器
    preprocessor = WineDataPreprocessorNoOutliers()

    # 執行前處理流程
    preprocessor.load_data()
    preprocessor.explore_data()
    preprocessor.clean_data()

    # 準備特徵和目標變數 (不新增特徵)
    X, y = preprocessor.prepare_features_target()

    # 分割資料
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    # 特徵標準化
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(
        X_train, X_val, X_test
    )

    # 儲存處理後資料
    preprocessor.save_processed_data(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    )

    # 生成報告
    preprocessor.generate_report()

    print("\n資料前處理完成 (保留異常值版本)！")
    return preprocessor


if __name__ == "__main__":
    preprocessor = main()
