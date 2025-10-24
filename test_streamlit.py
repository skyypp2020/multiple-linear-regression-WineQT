#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試Streamlit應用程式的基本功能
"""

import os
import sys


def test_imports():
    """測試所有必要的套件是否能正確匯入"""
    print("測試套件匯入...")

    try:
        import streamlit as st

        print("OK Streamlit 匯入成功")
    except ImportError as e:
        print(f"X Streamlit 匯入失敗: {e}")
        return False

    try:
        import pandas as pd

        print("✓ Pandas 匯入成功")
    except ImportError as e:
        print(f"✗ Pandas 匯入失敗: {e}")
        return False

    try:
        import numpy as np

        print("✓ NumPy 匯入成功")
    except ImportError as e:
        print(f"✗ NumPy 匯入失敗: {e}")
        return False

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        print("✓ Plotly 匯入成功")
    except ImportError as e:
        print(f"✗ Plotly 匯入失敗: {e}")
        return False

    try:
        import matplotlib.pyplot as plt

        print("✓ Matplotlib 匯入成功")
    except ImportError as e:
        print(f"✗ Matplotlib 匯入失敗: {e}")
        return False

    try:
        import seaborn as sns

        print("✓ Seaborn 匯入成功")
    except ImportError as e:
        print(f"✗ Seaborn 匯入失敗: {e}")
        return False

    try:
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

        print("✓ Scikit-learn 匯入成功")
    except ImportError as e:
        print(f"✗ Scikit-learn 匯入失敗: {e}")
        return False

    try:
        import joblib

        print("✓ Joblib 匯入成功")
    except ImportError as e:
        print(f"✗ Joblib 匯入失敗: {e}")
        return False

    return True


def test_data_files():
    """測試必要的資料檔案是否存在"""
    print("\n測試資料檔案...")

    required_files = [
        "datasets/WineQT.csv",
        "processed_data_no_outliers/X_test.csv",
        "processed_data_no_outliers/y_test.csv",
        "results_no_outliers/models/ridge_model.pkl",
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在")
        else:
            print(f"✗ {file_path} 不存在")
            all_exist = False

    return all_exist


def test_streamlit_app():
    """測試Streamlit應用程式是否能正確載入"""
    print("\n測試Streamlit應用程式...")

    try:
        # 嘗試匯入Streamlit應用程式
        sys.path.append(".")
        from streamlit_app import StreamlitApp

        app = StreamlitApp()
        print("✓ Streamlit應用程式類別載入成功")

        # 測試資料載入
        if app.load_data():
            print("✓ 資料載入成功")
        else:
            print("✗ 資料載入失敗")
            return False

        # 測試模型載入
        if app.load_models():
            print("✓ 模型載入成功")
        else:
            print("✗ 模型載入失敗")
            return False

        return True

    except Exception as e:
        print(f"✗ Streamlit應用程式測試失敗: {e}")
        return False


def main():
    """主測試函數"""
    print("=== Streamlit應用程式測試 ===")
    print()

    # 測試套件匯入
    if not test_imports():
        print("\n❌ 套件匯入測試失敗")
        return False

    # 測試資料檔案
    if not test_data_files():
        print("\n❌ 資料檔案測試失敗")
        return False

    # 測試Streamlit應用程式
    if not test_streamlit_app():
        print("\n❌ Streamlit應用程式測試失敗")
        return False

    print("\n✅ 所有測試通過！Streamlit應用程式可以正常運行。")
    print("\n啟動指令:")
    print("streamlit run streamlit_app.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
