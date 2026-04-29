import pandas as pd
import numpy as np

# 模拟数据验证IC和分位数收益的一致性
np.random.seed(42)

# 创建模拟数据：composite_score和fwd_return正相关
n = 500
composite_score = np.random.randn(n)
fwd_return = composite_score * 0.1 + np.random.randn(n) * 0.5  # 正相关但噪声大

df = pd.DataFrame({
    'composite_score': composite_score,
    'fwd_return': fwd_return
})

# 计算rank
df['rank'] = df['composite_score'].rank(ascending=False, method='first')

# 计算IC
def _spearman_rank_ic(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 30:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float) + 1
    rb = np.argsort(np.argsort(b)).astype(float) + 1
    ra_mean = np.mean(ra); rb_mean = np.mean(rb)
    num = np.sum((ra - ra_mean) * (rb - rb_mean))
    den = np.sqrt(np.sum((ra - ra_mean) ** 2) * np.sum((rb - rb_mean) ** 2))
    return num / den if den > 0 else np.nan

ic = _spearman_rank_ic(df['composite_score'].values, df['fwd_return'].values)
print(f"IC: {ic:.4f}")

# 计算分位数收益
print("\n分位数收益:")
for q in range(1, 11):
    lo_pct = (q - 1) * 10
    hi_pct = q * 10
    lo_rank = max(1, int(len(df) * lo_pct / 100))
    hi_rank = max(1, int(len(df) * hi_pct / 100))
    bucket = df[(df['rank'] >= lo_rank) & (df['rank'] <= hi_rank)]
    ret = bucket['fwd_return'].mean()
    print(f"Q{q:>2} ({lo_pct:>2}%-{hi_pct:>2}%) | 平均收益: {ret:>+7.4f}")

print("\n结论: 如果IC为正，Q1应该收益最高")
