import sqlite3
import pandas as pd
import numpy as np

# 从SQLite加载数据
conn = sqlite3.connect('stock_kline_cache.db')
df = pd.read_sql_query("SELECT date, pctChg FROM daily_kline", conn)
conn.close()

df['date'] = pd.to_datetime(df['date'])

print('df 日期范围:', df['date'].min(), '到', df['date'].max())
print('df 总条数:', len(df))

# 计算每日等权平均涨跌幅
daily_returns = df.groupby('date')['pctChg'].mean() / 100.0
daily_returns = daily_returns.dropna()

print('daily_returns 日期范围:', daily_returns.index.min(), '到', daily_returns.index.max())
print('daily_returns 数量:', len(daily_returns))

# 去掉极端值
daily_returns_clipped = daily_returns.clip(-0.20, 0.20)

# 计算累计收益率
log_returns = np.log1p(daily_returns_clipped)
cum_return = np.expm1(log_returns.sum())
print(f'累计收益率（全部数据）: {cum_return*100:.2f}%')

# 只计算2023-2025的数据
mask = (daily_returns.index >= '2023-01-01') & (daily_returns.index <= '2025-12-31')
daily_2023_2025 = daily_returns[mask]
daily_2023_2025_clipped = daily_2023_2025.clip(-0.20, 0.20)
log_returns_2023 = np.log1p(daily_2023_2025_clipped)
cum_return_2023 = np.expm1(log_returns_2023.sum())
print(f'累计收益率（2023-2025）: {cum_return_2023*100:.2f}%')
print(f'交易日数（2023-2025）: {len(daily_2023_2025)}')
