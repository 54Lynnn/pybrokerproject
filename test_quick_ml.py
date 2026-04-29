# -*- coding: utf-8 -*-
"""快速测试滚动XGBoost是否正常工作"""

import pandas as pd
import numpy as np
from factors import compute_factor_scores

# 创建测试数据
dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
np.random.seed(42)
data = []
for s in ['A', 'B', 'C', 'D', 'E']:
    for d in dates:
        data.append({
            'date': d, 'symbol': s, 'close': 100 + np.random.randn(),
            'f_reversal_5': np.random.randn(),
            'f_momentum_60': np.random.randn(),
            'f_volatility_20_inv': np.random.randn(),
        })
df = pd.DataFrame(data)

print('Testing rolling XGBoost...')
result = compute_factor_scores(df, use_rolling_ml=True)
print(f'ml_score exists: {"ml_score" in result.columns}')
if 'ml_score' in result.columns:
    print(f'ml_score valid count: {result["ml_score"].notna().sum()}')
    print(f'composite_score mean: {result["composite_score"].mean():.4f}')
else:
    print('ml_score not found!')
