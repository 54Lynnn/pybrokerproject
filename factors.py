# -*- coding: utf-8 -*-
"""多因子打分与选股"""

from config import Config
import numpy as np
import pandas as pd

def compute_factor_scores(df):
    """
    计算每个交易日每只股票的综合因子得分（7因子纯技术体系）。

    因子体系：
      Factor 1 - 20日动量 (momentum_20d)   — 强势股得分高
      Factor 2 - 量比 (volume_ratio)       — 放量股得分高
      Factor 3 - RSI得分 (rsi_score)       — RSI接近超卖区得分高（捕捉反转）
      Factor 4 - MACD得分 (macd_score)     — DIF>DEA得分高（金叉区域）
      Factor 5 - KDJ得分 (kdj_score)       — J值在20-80之间得分高（避免极端）
      Factor 6 - 布林带得分 (bb_score)     — %b在0.2-0.8之间得分高（非极端）
      Factor 7 - ATR得分 (atr_score)       — ATR较低得分高（低波动、回撤小）

    处理流程：
      1. 计算原始因子值
      2. 截面标准化（z-score）
      3. 加权求和 → 综合得分
      4. 每日排名 → selected=1
    """
    print(f"\n[因子] 正在计算多因子综合得分...")

    df = df.copy()

    # ===== Step 1: 计算原始因子值 =====

    # Factor 1: 20日动量 — IC=-0.033，方向翻转为均值回归
    # 高动量股在后续5日跑输 → 跌得多、近期弱势股反而有反弹空间
    df['f_momentum_20d'] = -df['return_20d']

    # Factor 2: 量比 — IC=-0.011，方向翻转
    # 放量股后续偏弱 → 缩量低换手的票更安全
    df['f_volume_ratio'] = -df['volume_ratio'].clip(0.1, 5.0)

    # Factor 3: RSI 得分 — 峰值在 RSI=35（超卖反转），RSI>70 得分低
    # RSI 原理：RSI<30=超卖（价格低估，可能反弹），RSI>70=超买（涨过头）
    # 峰值设在 35 而非 30，因为 RSI=35 时"接近超卖但未极端"，胜率更高
    rsi = df['rsi'].fillna(50).clip(0, 100).values
    df['f_rsi_score'] = 100 - ((rsi - 35) ** 2) / 200
    df['f_rsi_score'] = df['f_rsi_score'].clip(0, 100)

    # Factor 4: MACD 得分 — IC=-0.021，方向翻转
    # 金叉(DIF>DEA)反而跑输 → 死叉/低位股有回归空间，DIF低于DEA得分高
    dif = df['macd_dif'].fillna(0).values
    dea = df['macd_dea'].fillna(0).values
    close_vals = df['close'].values
    raw_macd = (dea - dif) / (close_vals + 0.01)   # 翻转: DEA > DIF 得分高
    df['f_macd_score'] = 100 / (1 + np.exp(-200 * raw_macd))

    # Factor 5: KDJ 得分 — K/D/J 均 < 20 且 K>D（偏离点金叉）→ 满分
    k_vals = df['kdj_k'].fillna(50).values
    d_vals = df['kdj_d'].fillna(50).values
    j_vals = df['kdj_j'].fillna(50).values
    # 三个条件：①三个值都低越好 ②K>D(金叉)+分 ③J不能钝化太久
    kdj_low = np.maximum(0, 20 - np.minimum(np.minimum(k_vals, d_vals), j_vals)) / 20  # 0~1
    kdj_cross = np.where(k_vals > d_vals, 0.5, 0)  # 金叉 +0.5
    kdj_j_ok = np.where((j_vals > 0) & (j_vals < 80), 0.3, -0.2)  # J不离群
    df['f_kdj_score'] = (kdj_low + kdj_cross + kdj_j_ok) * 100
    df['f_kdj_score'] = df['f_kdj_score'].clip(0, 100)

    # Factor 6: 布林带得分 — 收盘价跌破下轨（超跌反弹机会）得分高
    bb = df['bb_pct_b'].fillna(0.5).clip(-0.5, 1.5).values
    # %b < 0 = 跌破下轨 → 超跌 → 得分高；%b > 0.5 → 涨过中轨 → 0分
    df['f_bb_score'] = np.where(bb < 0.5, (0.5 - bb) * 100, 0)
    df['f_bb_score'] = df['f_bb_score'].clip(0, 100)

    # Factor 7: ATR 得分 — 低波动得分高
    # ATR 原理：衡量股票的真实波动幅度（含跳空）。ATR 高 = 剧烈波动 = 风险大
    # ATR 低 = 温和调整 = 更适合轮动入场
    atr = df['atr'].fillna(0).values
    df['f_atr_score'] = np.where(atr > 0, 1.0 / (atr / close_vals + 0.01), 0)

    # Factor 8: 5日动量 — 短期追涨（与20日反转正交，牛市适用）
    df['f_momentum_5d'] = df['return_5d']

    # Factor 9: CCI — 趋势强度，极端负值回归
    cci = df['cci'].fillna(0).clip(-300, 300).values
    df['f_cci_score'] = -np.abs(cci)  # 负CCi(超卖)回归得分高，正CCI不追

    # ===== Step 2: 截面标准化 =====
    factor_cols = ['f_momentum_20d', 'f_momentum_5d', 'f_volume_ratio', 'f_rsi_score',
                   'f_macd_score', 'f_kdj_score', 'f_bb_score', 'f_cci_score', 'f_atr_score']

    # 因子平滑：每只股票每个因子做 5 日滚动均值，消除日间噪声
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    for col in factor_cols:
        df[col] = df.groupby('symbol')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    print(f"\n  正在进行截面标准化（{df['date'].nunique()} 个交易日）...")
    result_list = []
    dates = sorted(df['date'].unique())
    for i, d in enumerate(dates):
        mask = df['date'] == d
        group = df[mask].copy()
        for col in factor_cols:
            std = group[col].std()
            if std == 0 or np.isnan(std):
                group[col] = 0.0
            else:
                group[col] = (group[col] - group[col].mean()) / std
        result_list.append(group)
        if (i + 1) % 100 == 0 or (i + 1) == len(dates):
            print(f"  标准化进度: {i+1}/{len(dates)} ({100*(i+1)/len(dates):.0f}%)")

    df = pd.concat(result_list, ignore_index=True)

    # ===== Step 3: 加权求和 =====
    weights = Config.FACTOR_WEIGHTS
    df['composite_score'] = 0.0
    for key, w in weights.items():
        col = 'f_' + key
        if col in df.columns:
            df['composite_score'] += df[col].fillna(0) * w

    # ===== Step 4: 排名 =====
    df['rank'] = df.groupby('date')['composite_score'].rank(ascending=False, method='first')

    # ===== Step 5: 选股信号 =====
    df['selected'] = np.where(df['rank'] <= Config.SELL_THRESHOLD, 1, 0)

    print(f"  ✓ 因子得分计算完成，每日选取前 {Config.TOP_N_STOCKS} 只（卖出阈值: 前 {Config.SELL_THRESHOLD} 只）")
    return df


SELECTION_MAP = {}

def build_daily_selections(df):
    """从因子打分结果构建每日选股映射表。"""
    selected_df = df[df['selected'] == 1][['date', 'symbol']].copy()
    selected_df['date'] = pd.to_datetime(selected_df['date'])
    selection_map = {}
    for date, group in selected_df.groupby('date'):
        selection_map[date] = set(group['symbol'].tolist())
    print(f"  ✓ 构建每日选股表：{len(selection_map)} 个交易日")
    return selection_map
