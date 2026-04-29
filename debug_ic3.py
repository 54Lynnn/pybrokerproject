import sys
sys.path.insert(0, r'E:\project\pybroker')

import pandas as pd
import numpy as np
from factor_analysis import compute_forward_return, _spearman_rank_ic

# 手动构建一个简化的数据集来验证
# 使用实际回测中的逻辑

# 模拟5只股票，3天的数据
data = []
for day in range(3):
    for sym in ['A', 'B', 'C', 'D', 'E']:
        data.append({
            'symbol': sym,
            'date': pd.Timestamp(f'2023-01-{day+1}'),
            'close': 10 + np.random.randn(),
            'composite_score': np.random.randn()
        })

df = pd.DataFrame(data)
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

# 计算rank
df['rank'] = df.groupby('date')['composite_score'].rank(ascending=False, method='first')

# 计算未来收益
df = compute_forward_return(df, 1)

print("原始数据:")
print(df[['symbol', 'date', 'composite_score', 'rank', 'fwd_return']])

# 按日期分组计算IC和分位数收益
for d, grp in df.groupby('date'):
    print(f"\n日期: {d}")
    ic = _spearman_rank_ic(grp['composite_score'].values, grp['fwd_return'].values)
    print(f"IC: {ic:.4f}")
    
    # 分位数收益
    for q in range(1, 3):  # 只分2组
        lo_pct = (q - 1) * 50
        hi_pct = q * 50
        lo_rank = max(1, int(len(grp) * lo_pct / 100))
        hi_rank = max(1, int(len(grp) * hi_pct / 100))
        bucket = grp[(grp['rank'] >= lo_rank) & (grp['rank'] <= hi_rank)]
        ret = bucket['fwd_return'].mean()
        print(f"Q{q} | rank {lo_rank}-{hi_rank} | 平均收益: {ret:+.4f}")
