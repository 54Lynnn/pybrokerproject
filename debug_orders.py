import sqlite3
import pandas as pd
import numpy as np

# 从SQLite加载数据
conn = sqlite3.connect('stock_kline_cache.db')
df = pd.read_sql_query("""
    SELECT code, date, pctChg
    FROM daily_kline
    WHERE date >= '2023-01-01' AND date <= '2025-12-31'
    ORDER BY code, date
""", conn)
conn.close()

df['date'] = pd.to_datetime(df['date'])
df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')

# 计算每日等权平均涨跌幅
daily_returns = df.groupby('date')['pctChg'].mean() /