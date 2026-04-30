# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import sqlite3
from config import Config

# 读取数据
db_path = Config.SQLITE_DB_PATH
conn = sqlite3.connect(db_path)

# 获取所有股票代码
codes = pd.read_sql('SELECT DISTINCT code FROM daily_kline LIMIT 100', conn)
print(f'数据库中股票数量: {len(codes)}')

# 读取最近的数据
df = pd.read_sql("SELECT * FROM daily_kline WHERE date >= '2023-01-01' AND date <= '2025-12-31' LIMIT 50000", conn)

print(f'数据行数: {len(df)}')
print(f'列: {list(df.columns)}')
print('日期范围:', df['date'].min(), '~', df['date'].max())

conn.close()
