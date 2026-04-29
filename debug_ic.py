import sqlite3
import pandas as pd
import numpy as np

# 从数据库加载已计算的因子数据
db_path = r'E:\project\pybroker\stock_kline_cache.db'
conn = sqlite3.connect(db_path)

# 加载股票数据
query = """
SELECT code, date, close FROM daily_kline
WHERE date >= '2023-01-01' AND date <= '2025-12-31'
ORDER BY code, date
"""
df = pd.read_sql_query(query, conn)
conn.close()

df.columns = ['symbol', 'date', 'close']
df['date'] = pd.to_datetime(df['date'], format='mixed')
df['close'] = pd.to_numeric(df['close'], errors='coerce')

# 计算未来5日收益
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
df['fwd_close'] = df.groupby('symbol')['close'].shift(-5)
df['fwd_return'] = df['fwd_close'] / df['close'] - 1

# 计算简单的低波动因子（1/ATR）
# 用20日收益率标准差作为波动率代理
df['daily_return'] = df.groupby('symbol')['close'].pct_change()
df['volatility'] = df.groupby('symbol')['daily_return'].rolling(window=20, min_periods=5).std().reset_index(0, drop=True)
df['factor'] = 1.0 / (df['volatility'] + 0.01)

# 截面标准化
df['factor_z'] = df.groupby('date')['factor'].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

# 计算IC
daily_ics = []
for d, grp in df.groupby('date'):
    grp = grp.dropna(subset=['factor_z', 'fwd_return'])
    if len(grp) > 30:
        # Spearman IC
        from scipy.stats import spearmanr
        ic, pval = spearmanr(grp['factor_z'], grp['fwd_return'])
        daily_ics.append(ic)

ics = pd.Series(daily_ics)
print(f"IC均值: {ics.mean():.4f}")
print(f"IC中位数: {ics.median():.4f}")
print(f"IC>0比例: {(ics > 0).mean()*100:.1f}%")

# 检查分位数收益
df['rank'] = df.groupby('date')['factor_z'].rank(ascending=False)

print("\n分位数收益（factor_z排名）:")
for q in range(1, 11):
    lo = (q-1) * 10
    hi = q * 10
    returns = []
    for d, grp in df.groupby('date'):
        n = len(grp)
        lo_rank = max(1, int(n * lo / 100))
        hi_rank = max(1, int(n * hi / 100))
        bucket = grp[(grp['rank'] >= lo_rank) & (grp['rank'] <= hi_rank)]
        ret = bucket['fwd_return'].mean()
        if not np.isnan(ret):
            returns.append(ret)
    if returns:
        daily = np.mean(returns)
        cum = np.prod([1+r for r in returns]) - 1
        print(f"Q{q:>2} ({lo:>2}%-{hi:>2}%) | 日均: {daily:>+7.4f} | 累计: {cum:>+8.2%}")
