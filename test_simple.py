#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡單的Streamlit應用程式測試
"""

import os
import sys


def test_basic_imports():
    """測試基本套件匯入"""
    print("測試基本套件匯入...")

    try:
        import streamlit

        print("OK Streamlit 可用")
    except ImportError:
        print("X Streamlit 不可用")
        return False

    try:
        import pandas

        print("OK Pandas 可用")
    except ImportError:
        print("X Pandas 不可用")
        return False

    try:
        import numpy

        print("OK NumPy 可用")
    except ImportError:
        print("X NumPy 不可用")
        return False

    try:
        import plotly

        print("OK Plotly 可用")
    except ImportError:
        print("X Plotly 不可用")
        return False

    return True


def test_files():
    """測試必要檔案"""
    print("\n測試必要檔案...")

    files_to_check = ["streamlit_app.py", "datasets/WineQT.csv"]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"OK {file_path} 存在")
        else:
            print(f"X {file_path} 不存在")
            return False

    return True


def main():
    """主測試函數"""
    print("=== Streamlit 應用程式基本測試 ===")

    if not test_basic_imports():
        print("\n測試失敗: 套件匯入問題")
        return False

    if not test_files():
        print("\n測試失敗: 檔案不存在")
        return False

    print("\n基本測試通過!")
    print("可以嘗試啟動 Streamlit 應用程式:")
    print("streamlit run streamlit_app.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
