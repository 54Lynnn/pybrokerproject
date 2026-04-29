import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('stock_kline_cache.db')
df = pd.read_sql_query("SELECT date, pctChg FROM daily_kline WHERE date >= '2023-01-01' AND date <= '2025-12-31'", conn)
conn.close()

print('数据条数:', len(df))
print('pctChg 统计:')
print(df['pctChg'].describe())
print()

# 按日期分组计算等权平均
daily = df.groupby('date')['pctChg'].mean() / 100.0
print('日平均收益统计:')
print(daily.describe())
print()

# 计算累计收益
log_returns = np.log1p(daily)
cum_return = np.expm1(log_returns.sum())
print(f'累计收益率: {cum_return*100:.2f}%')
print(f'交易日数: {len(daily)}')

# 检查是否有异常值
print()
print('日收益前10大:')
print(daily.nlargest(10))
print()
print('日收益前10小:')
print(daily.nsmallest(10))
