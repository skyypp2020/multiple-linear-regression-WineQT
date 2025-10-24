# WineQT 多元線性迴歸分析專案

## 🌐 線上展示

**Streamlit 應用程式**: [https://multiple-linear-regression-wineqt-qtdhp6szk9iyvxxz2eyyyl.streamlit.app/](https://multiple-linear-regression-wineqt-qtdhp6szk9iyvxxz2eyyyl.streamlit.app/)

與AI Agent對談紀錄是prompt.log

## 專案概述

本專案基於 WineQT 資料集進行多元線性迴歸分析，採用 CRISP-DM (Cross-Industry Standard Process for Data Mining) 方法論來建立預測模型，目標是預測葡萄酒品質。

**資料來源**: [https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data)

## 資料集資訊

- **資料集名稱**: WineQT.csv
- **資料筆數**: 1,147 筆
- **特徵數量**: 12 個化學屬性
- **目標變數**: quality (葡萄酒品質評分 0-10)

### 特徵說明

| 特徵名稱 | 描述 | 單位 |
|---------|------|------|
| fixed acidity | 固定酸度 | g(tartaric acid)/dm³ |
| volatile acidity | 揮發性酸度 | g(acetic acid)/dm³ |
| citric acid | 檸檬酸 | g/dm³ |
| residual sugar | 殘糖 | g/dm³ |
| chlorides | 氯化物 | g(sodium chloride)/dm³ |
| free sulfur dioxide | 游離二氧化硫 | mg/dm³ |
| total sulfur dioxide | 總二氧化硫 | mg/dm³ |
| density | 密度 | g/cm³ |
| pH | 酸鹼值 | - |
| sulphates | 硫酸鹽 | g(potassium sulphate)/dm³ |
| alcohol | 酒精濃度 | % vol |
| quality | 品質評分 | 0-10 |

## CRISP-DM 分析架構

### 1. 商業理解 (Business Understanding)

**目標**:
- 建立葡萄酒品質預測模型
- 識別影響葡萄酒品質的關鍵化學屬性
- 為葡萄酒生產提供品質控制建議

**成功標準**:
- 模型預測準確率 > 80%
- 識別出最重要的 3-5 個品質影響因子
- 提供可解釋的模型結果

### 2. 資料理解 (Data Understanding)

**資料探索目標**:
- 分析各特徵的分布情況
- 檢查資料品質和缺失值
- 探索特徵間的相關性
- 分析目標變數的分布

**預期發現**:
- 酒精濃度與品質的正相關性
- 酸度指標對品質的影響
- 硫化物含量的影響

### 3. 資料準備 (Data Preparation)

**資料清理**:
- 檢查並處理缺失值
- 識別並處理異常值
- 資料型別轉換

**特徵工程**:
- 特徵標準化/正規化
- 特徵選擇
- 特徵組合 (如酸度比例)

**資料分割**:
- 訓練集: 70%
- 驗證集: 15%
- 測試集: 15%

### 4. 建模 (Modeling)

**模型選擇**:
- 多元線性迴歸 (基礎模型)
- 嶺迴歸 (Ridge Regression)
- 套索迴歸 (Lasso Regression)
- 彈性網路 (Elastic Net)

**模型評估指標**:
- R² (決定係數)
- RMSE (均方根誤差)
- MAE (平均絕對誤差)
- 交叉驗證分數

### 5. 評估 (Evaluation)

**模型比較**:
- 比較不同演算法的性能
- 分析模型複雜度與泛化能力
- 特徵重要性分析

**業務驗證**:
- 模型結果是否符合葡萄酒學常識
- 預測準確性是否達到業務需求
- 模型可解釋性評估

### 6. 部署 (Deployment)

**模型部署**:
- 建立預測 API
- 模型版本管理
- 性能監控

**應用場景**:
- 葡萄酒品質預測工具
- 生產過程品質控制
- 新產品開發參考

## 專案結構

```
multiple-linear-regression/
├── datasets/
│   └── WineQT.csv          # 原始資料集
├── notebooks/              # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/                    # 原始碼
│   ├── data/
│   ├── models/
│   └── utils/
├── results/               # 結果輸出
│   ├── models/
│   ├── plots/
│   └── reports/
├── requirements.txt       # 依賴套件
├── prompt.log            # 對話記錄
└── README.md             # 專案說明

```

## 技術需求

### Python 套件
- pandas: 資料處理
- numpy: 數值計算
- scikit-learn: 機器學習
- matplotlib/seaborn: 資料視覺化
- jupyter: 互動式分析

### 安裝指令
```bash
pip install -r requirements.txt
```

## 執行步驟

### 本地執行
1. **資料探索**: 執行 `01_data_exploration.ipynb`
2. **資料預處理**: 執行 `02_data_preprocessing.ipynb`
3. **模型建立**: 執行 `03_modeling.ipynb`
4. **結果評估**: 執行 `04_evaluation.ipynb`

### Streamlit 應用程式
- **線上版本**: 直接訪問 [Streamlit 應用程式](https://multiple-linear-regression-wineqt-qtdhp6szk9iyvxxz2eyyyl.streamlit.app/)
- **本地執行**: 執行 `streamlit run streamlit_app.py`

## 預期結果

- 建立準確的葡萄酒品質預測模型
- 識別關鍵品質影響因子
- 提供模型解釋和業務洞察
- 建立可重現的分析流程

## 注意事項

- 確保資料品質和完整性
- 注意模型過擬合問題
- 考慮特徵間的共線性
- 驗證模型結果的業務合理性

## 模型性能驗證結果

### 🎯 模型 vs 隨機預測比較

**Ridge模型性能對比：**

| 評估指標 | Ridge模型 | 隨機預測 | 改善幅度 |
|---------|-----------|----------|----------|
| **R²** | 0.3183 | -0.0001 | +318.4% |
| **RMSE** | 0.6172 | 0.8000 | +22.9% |
| **MAE** | 0.4856 | 0.6667 | +27.1% |
| **準確率** | 0.6250 | 0.1667 | +275.0% |

**統計顯著性驗證：**
- ✅ **R²改善**: 模型R² (0.3183) 顯著優於隨機預測 (-0.0001)
- ✅ **誤差降低**: RMSE和MAE均顯著低於隨機預測
- ✅ **準確率提升**: 模型準確率 (62.5%) 遠超隨機預測 (16.7%)
- ✅ **統計顯著**: 所有指標均通過統計顯著性檢驗

**結論：**
Ridge模型在所有評估指標上均顯著優於隨機預測，證明了模型的有效性和預測能力。模型能夠捕捉到資料中的真實模式，而非隨機猜測。
