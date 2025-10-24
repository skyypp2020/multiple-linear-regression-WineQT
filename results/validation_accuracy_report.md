
# WineQT 驗證集準確率分析報告

## 概述
- 分析時間: 2025-10-24 21:01:56
- 驗證集樣本數: 206 筆
- 測試集樣本數: 206 筆
- 分析模型數: 8 個

## 驗證集準確率排名

| 排名 | 模型 | 驗證集準確率 | 測試集準確率 | 準確率差異 |
|------|------|-------------|-------------|-----------|
| 1 | Baseline_LinearRegression | 0.6117 | 0.6019 | -0.0097 |
| 2 | Baseline_Ridge | 0.6068 | 0.6019 | -0.0049 |
| 3 | Selected_LinearRegression | 0.6068 | 0.6019 | -0.0049 |
| 4 | Selected_Ridge | 0.6068 | 0.5922 | -0.0146 |
| 5 | Baseline_Lasso | 0.5922 | 0.6408 | 0.0485 |
| 6 | Baseline_ElasticNet | 0.5922 | 0.6019 | 0.0097 |
| 7 | Selected_Lasso | 0.5922 | 0.6408 | 0.0485 |
| 8 | Selected_ElasticNet | 0.5922 | 0.5971 | 0.0049 |

## 最佳模型: Baseline_LinearRegression

### 性能指標
- 驗證集準確率: 0.6117 (61.17%)
- 測試集準確率: 0.6019 (60.19%)
- 準確率差異: -0.0097

## 模型類型分析

### Baseline 模型
- Baseline_LinearRegression: 0.6117 (61.17%)
- Baseline_Ridge: 0.6068 (60.68%)
- Baseline_Lasso: 0.5922 (59.22%)
- Baseline_ElasticNet: 0.5922 (59.22%)

### 特徵選擇後模型
- Selected_LinearRegression: 0.6068 (60.68%)
- Selected_Ridge: 0.6068 (60.68%)
- Selected_Lasso: 0.5922 (59.22%)
- Selected_ElasticNet: 0.5922 (59.22%)

## 結論

1. **最佳模型**: Baseline_LinearRegression 在驗證集上表現最佳
2. **模型穩定性**: 驗證集與測試集準確率差異分析
3. **特徵選擇效果**: 比較 Baseline 和特徵選擇後模型的性能
4. **模型選擇建議**: 基於驗證集性能選擇最佳模型

## 建議

1. **模型部署**: 使用 Baseline_LinearRegression 作為最終模型
2. **性能監控**: 持續監控模型在生產環境中的表現
3. **模型優化**: 基於驗證集結果進一步優化模型參數
4. **集成學習**: 考慮結合多個表現良好的模型
