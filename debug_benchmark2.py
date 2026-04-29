import sqlite3
import pandas as pd
import numpy as np

# 从SQLite加载数据
conn = sqlite3.connect('stock_kline_cache.db')
df = pd.read_sql_query("SELECT date, pctChg FROM daily_kline WHERE date >= '2023-01-01' AND date <= '2025-12-31'", conn)
conn.close()

df['date'] = pd.to_datetime(df['date'])

# 按日期分组计算等权平均涨跌幅
daily = df.groupby('date')['pctChg'].mean()
print('日平均收益前20:')
print(daily.nlargest(20))
print()
print('日平均收益后20:')
print(daily.nsmallest(20))
print()

# 检查是否有某一天的平均值异常
print('日平均收益统计:')
print(daily.describe())
