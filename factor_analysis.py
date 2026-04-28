# -*- coding: utf-8 -*-
"""因子评估: IC 分析 + 分位数收益 + 衰减曲线"""

from config import Config
import numpy as np
import pandas as pd

def compute_forward_return(df, horizon_days):
    """向量化计算每只股票未来 N 日收益率（按月分组+shift）。"""
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    df['fwd_close'] = df.groupby('symbol')['close'].shift(-horizon_days)
    df['fwd_return'] = df['fwd_close'] / df['close'] - 1
    df.drop(columns=['fwd_close'], inplace=True)
    return df


def _spearman_rank_ic(a, b):
    """纯 NumPy Spearman rank IC（避免 scipy Fortran crash）。"""
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


def evaluate_factors(df, horizons=(1, 5, 10)):
    """
    因子评估主函数。

    输出：
      1. 综合得分 IC（日频，每个 horizon）
      2. 每个因子的 IC
      3. 分位数组合收益（Q1=排名前10%, Q10=排名后10%）
      4. 因子自相关衰减（信号持续性）

    Args:
        df: 包含 composite_score / rank / 各 f_* 列的 DataFrame
        horizons: 前向收益率计算周期（交易日）
    """
    print(f"\n{'=' * 60}")
    print(f"  因子评估报告")
    print(f"{'=' * 60}")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # ===== 1. 综合得分 IC =====
    print(f"\n  [1] 综合得分 Rank IC（Spearman）")
    print(f"       周期     |   均值  |  中位数 |  IR   |  正>0比例")
    print(f"  " + "-" * 56)

    for h in horizons:
        df_with_fwd = compute_forward_return(df, h)
        daily_ics = []
        for d, grp in df_with_fwd.groupby('date'):
            ic = _spearman_rank_ic(grp['composite_score'], grp['fwd_return'])
            if not np.isnan(ic):
                daily_ics.append(ic)
        if daily_ics:
            ics = pd.Series(daily_ics)
            mean_ic = ics.mean()
            med_ic = ics.median()
            ir = mean_ic / ics.std() if ics.std() > 0 else 0
            pos_pct = (ics > 0).mean() * 100
            sign = '+' if mean_ic > 0 else ''
            print(f"     forward {h:>2}d    |  {sign}{mean_ic:>6.4f} | {med_ic:>7.4f} | {ir:>6.3f} | {pos_pct:>5.1f}%")

    # ===== 2. 各因子 IC =====
    factor_names = {
        'f_momentum_20d': '动量', 'f_volume_ratio': '量比', 'f_rsi_score': 'RSI',
        'f_macd_score': 'MACD', 'f_kdj_score': 'KDJ', 'f_bb_score': 'BB',
        'f_atr_score': 'ATR'
    }
    print(f"\n  [2] 各因子 IC（forward 5d）")
    print(f"       因子    |   均值  |  正>0比例")
    print(f"  " + "-" * 40)

    df_fwd5 = compute_forward_return(df, 5)
    for col, label in factor_names.items():
        if col in df.columns:
            daily_ics = []
            for d, grp in df_fwd5.groupby('date'):
                ic = _spearman_rank_ic(grp[col], grp['fwd_return'])
                if not np.isnan(ic):
                    daily_ics.append(ic)
            if daily_ics:
                ics = pd.Series(daily_ics)
                mean_ic = ics.mean()
                pos_pct = (ics > 0).mean() * 100
                sign = '+' if mean_ic > 0 else ''
                print(f"     {label:<6s}   |  {sign}{mean_ic:>6.4f} | {pos_pct:>5.1f}%")

    # ===== 3. 分位数收益 =====
    print(f"\n  [3] 分位数组合收益（Compositive Score 排名, forward 5d）")
    print(f"       分组    |  日均收益 |  累计收益")
    print(f"  " + "-" * 45)

    df_fwd5 = compute_forward_return(df, 5)
    for q in range(1, 11):
        lo_pct = (q - 1) * 10
        hi_pct = q * 10
        returns = []
        for d, grp in df_fwd5.groupby('date'):
            lo_rank = max(1, int(len(grp) * lo_pct / 100))
            hi_rank = max(1, int(len(grp) * hi_pct / 100))
            bucket = grp[(grp['rank'] >= lo_rank) & (grp['rank'] <= hi_rank)]
            ret = bucket['fwd_return'].mean()
            if not np.isnan(ret):
                returns.append(ret)
        if returns:
            rets = pd.Series(returns)
            daily = rets.mean()
            cum = (1 + rets).prod() - 1
            marker = ' ◀ Q1' if q == 1 else (' ◀ Q10' if q == 10 else '')
            print(f"     Q{q:>2} ({lo_pct:>2}%-{hi_pct:>2}%) |  {daily:>+7.4f} | {cum:>+9.2%}{marker}")

    # ===== 4. 因子衰减 =====
    print(f"\n  [4] 综合得分自相关衰减（信号持续性）")
    print(f"       Lag  |  自相关")
    print(f"  " + "-" * 28)
    all_acf = []
    for sym, grp in df.groupby('symbol'):
        scores = grp['composite_score'].dropna().values
        if len(scores) > 20:
            for lag in range(1, min(21, len(scores))):
                if len(scores) > lag:
                    acf = np.corrcoef(scores[:-lag], scores[lag:])[0, 1]
                    if not np.isnan(acf):
                        all_acf.append((lag, acf))
    acf_df = pd.DataFrame(all_acf, columns=['lag', 'acf'])
    for lag in [1, 3, 5, 10, 20]:
        vals = acf_df[acf_df['lag'] == lag]['acf']
        if len(vals) > 0:
            print(f"       {lag:>3}d   |  {vals.mean():.4f}")

    half_life = None
    for lag in range(1, 21):
        vals = acf_df[acf_df['lag'] == lag]['acf']
        if len(vals) > 0 and vals.mean() < 0.5:
            half_life = lag
            break
    if half_life:
        print(f"\n    → 信号半衰期 ≈ {half_life} 天（自相关跌破 0.5）")
        print(f"    → 建议 MIN_HOLD_BARS 设为 {max(3, half_life)}")
        print(f"    → 建议 SELL_THRESHOLD 设为 TOP_N × {max(2, half_life//3 + 1)}")

    print(f"\n{'=' * 60}")
