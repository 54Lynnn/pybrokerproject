import sqlite3
import pandas as pd
import numpy as np

# 模拟 main.py 中的数据加载
conn = sqlite3.connect('stock_kline_cache.db')
df = pd.read_sql_query("""
    SELECT code, date, open, high, low, close, volume, amount, turn, pctChg,
           peTTM, pbMRQ, psTTM, pcfNcfTTM
    FROM daily_kline
    WHERE date >= '2022-10-03' AND date <= '2025-12-31'
    ORDER BY code, date
""", conn)
conn.close()

df.rename(columns={'code': 'symbol'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])

numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[df['volume'] > 0].copy()

print('df 日期范围:', df['date'].min(), '到', df['date'].max())
print('df 总条数:', len(df))

# 计算每日等权平均涨跌幅
daily_returns = df.groupby('date')['pctChg'].mean() / 100.0
daily_returns = daily_returns.dropna()

print('daily_returns 日期范围:', daily_returns.index.min(), '到', daily_returns.index.max())
print('daily_returns 数量:', len(daily_returns))

# 限制单日涨跌幅
daily_returns = daily_returns.clip(-0.10, 0.10)

# 计算累计收益率
log_returns = np.log1p(daily_returns)
cum_return = np.expm1(log_returns.sum())
print(f'累计收益率: {cum_return*100:.2f}%')
