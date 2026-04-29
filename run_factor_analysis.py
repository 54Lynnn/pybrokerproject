# -*- coding: utf-8 -*-
"""运行完整的因子分析"""

import sys
sys.path.insert(0, 'e:\\project\\pybroker')

import pandas as pd
import numpy as np
import sqlite3
from config import Config
from indicators import compute_all_indicators
from factor_engineering import generate_factors, standardize_factors
from factor_analysis_report import run_full_factor_analysis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*60)
print("开始因子分析...")
print("="*60)

# 读取数据
db_path = Config.SQLITE_DB_PATH
conn = sqlite3.connect(db_path)

# 读取回测期间的数据
df = pd.read_sql("""
    SELECT * FROM daily_kline 
    WHERE date >= '2023-01-01' AND date <= '2025-12-31'
""", conn)

conn.close()

print(f"原始数据: {len(df)} 行, {df['code'].nunique()} 只股票")

# 重命名列以匹配策略
df = df.rename(columns={
    'code': 'symbol',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume'
})

# 转换数据类型
df['date'] = pd.to_datetime(df['date'], format='mixed')
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 删除缺失值
df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

print(f"清洗后数据: {len(df)} 行")

# 计算技术指标
print("\n计算技术指标...")
df = compute_all_indicators(df)

# 生成因子
print("\n生成因子...")
df, factor_names = generate_factors(df)

print(f"生成 {len(factor_names)} 个因子: {factor_names}")

# 标准化
print("\n标准化因子...")
df = standardize_factors(df, factor_names)

# 运行完整分析
print("\n运行因子分析...")
report = run_full_factor_analysis(df, factor_names)

print("\n分析完成!")
