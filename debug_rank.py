import pandas as pd
import numpy as np

# 创建模拟数据验证rank逻辑
df = pd.DataFrame({
    'symbol': ['A', 'B', 'C', 'D', 'E'],
    'composite_score': [10, 5, 8, 2, 6]
})

# 计算rank（和factors.py中一样的逻辑）
df['rank'] = df['composite_score'].rank(ascending=False, method='first')

print("模拟数据验证rank逻辑:")
print(df.sort_values('rank'))
print(f"\nRank 1 = 最高分: {df[df['rank']==1]['composite_score'].values[0]}")
print(f"Rank 5 = 最低分: {df[df['rank']==5]['composite_score'].values[0]}")
