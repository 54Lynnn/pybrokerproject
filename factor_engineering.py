# -*- coding: utf-8 -*-
"""
精选高IC因子 — 专业优化版（修复动量方向）

关键修复：
  1. A股短期（5-20日）是反转效应，应取负值
  2. A股中期（60日+）才是动量效应，取正值
  3. 增加短期反转因子
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def generate_factors(df):
    """
    生成经过专业优化的因子体系。
    
    A股特性：
      - 短期（1-20日）：反转效应（涨多了跌，跌多了涨）
      - 中期（60-120日）：动量效应（强者恒强）
      - 长期（240日+）：价值回归
    """
    logger.info("正在生成专业优化因子...")
    
    df = df.copy()
    close = df['close'].values
    
    factors = {}
    
    # ============================================================
    # 1. 短期反转因子（A股核心Alpha，1-20日）
    # ============================================================
    
    # 1.1 5日反转（最强反转效应）
    mom_5 = df.groupby('symbol')['close'].transform(
        lambda x: x / x.shift(5) - 1
    ).fillna(0).values
    # 短期涨多了要跌，所以取负值
    factors['f_reversal_5'] = np.clip(-mom_5, -0.3, 0.3)
    
    # 1.2 10日反转
    mom_10 = df.groupby('symbol')['close'].transform(
        lambda x: x / x.shift(10) - 1
    ).fillna(0).values
    factors['f_reversal_10'] = np.clip(-mom_10, -0.5, 0.5)
    
    # 1.3 20日反转
    mom_20 = df.groupby('symbol')['close'].transform(
        lambda x: x / x.shift(20) - 1
    ).fillna(0).values
    factors['f_reversal_20'] = np.clip(-mom_20, -0.5, 0.5)
    
    # ============================================================
    # 2. 中期动量因子（60日+，强者恒强）
    # ============================================================
    
    # 2.1 60日动量（中期趋势）
    mom_60 = df.groupby('symbol')['close'].transform(
        lambda x: x / x.shift(60) - 1
    ).fillna(0).values
    # 中期动量效应，涨得好的继续涨
    factors['f_momentum_60'] = np.clip(mom_60, -1.0, 1.0)
    
    # 2.2 120日动量（长期趋势）
    mom_120 = df.groupby('symbol')['close'].transform(
        lambda x: x / x.shift(120) - 1
    ).fillna(0).values
    factors['f_momentum_120'] = np.clip(mom_120, -1.5, 1.5)
    
    # ============================================================
    # 3. 趋势强度因子
    # ============================================================
    
    # 3.1 均线多头排列强度
    if 'ma_5' in df.columns and 'ma_20' in df.columns:
        ma5 = df['ma_5'].fillna(0).values
        ma20 = df['ma_20'].fillna(0).values
        # 价格在均线上方且短期均线上穿长期均线
        trend_score = np.where(
            (close > ma20) & (ma5 > ma20),
            (close / (ma20 + 1e-6) - 1) + (ma5 / (ma20 + 1e-6) - 1),
            0
        )
        factors['f_trend_strength'] = np.clip(trend_score, 0, 0.5)
    
    # 3.2 ADX趋势强度
    if 'adx' in df.columns:
        adx = df['adx'].fillna(0).values
        factors['f_adx_trend'] = np.where(adx > 25, (adx - 25) / 75, 0)
    
    # ============================================================
    # 4. 低波类因子（风险控制）
    # ============================================================
    
    # 4.1 多周期波动率倒数
    for window in [5, 20, 60]:
        vol = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window, min_periods=5).std()
        ).fillna(0.01).values
        vol_clean = np.clip(vol, 0.001, 0.5)
        factors[f'f_volatility_{window}_inv'] = 1.0 / vol_clean
    
    # 4.2 ATR倒数
    if 'atr' in df.columns:
        atr = df['atr'].fillna(0).values
        close_safe = np.where(close > 0, close, 1)
        atr_pct = atr / close_safe
        atr_pct_clean = np.clip(atr_pct, 0.001, 0.5)
        factors['f_atr_inv'] = 1.0 / atr_pct_clean
    
    # ============================================================
    # 5. 反转类因子（技术指标超卖）
    # ============================================================
    
    # 5.1 RSI超卖
    if 'rsi' in df.columns:
        rsi = df['rsi'].fillna(50).values
        factors['f_rsi_oversold'] = np.where(rsi < 35, (35 - rsi) / 35, 0)
        factors['f_rsi_overbought'] = np.where(rsi > 70, (rsi - 70) / 30, 0)
    
    # 5.2 WILLR超卖
    if 'willr' in df.columns:
        wr = df['willr'].fillna(-50).values
        factors['f_willr_oversold'] = np.where(wr < -80, (-wr - 80) / 20, 0)
    
    # 5.3 KDJ低位
    if 'kdj_j' in df.columns:
        j = df['kdj_j'].fillna(50).values
        factors['f_kdj_j_low'] = np.where(j < 0, -j / 100, 0)
        factors['f_kdj_j_high'] = np.where(j > 100, (j - 100) / 100, 0)
    
    # 5.4 综合反转信号
    if 'rsi' in df.columns and 'willr' in df.columns and 'kdj_j' in df.columns:
        rsi = df['rsi'].fillna(50).values
        wr = df['willr'].fillna(-50).values
        j = df['kdj_j'].fillna(50).values
        factors['f_combined_reversal'] = (
            (rsi < 35).astype(float) * 0.4 +
            (wr < -80).astype(float) * 0.3 +
            (j < 0).astype(float) * 0.3
        )
    
    # 5.5 布林带位置
    if 'bb_pct_b' in df.columns:
        bb = df['bb_pct_b'].fillna(0.5).values
        factors['f_bb_low'] = np.where(bb < 0.1, (0.1 - bb) / 0.1, 0)
        factors['f_bb_high'] = np.where(bb > 0.9, (bb - 0.9) / 0.1, 0)
    
    # ============================================================
    # 6. 估值类因子（长期Alpha）
    # ============================================================
    
    # 6.1 PB倒数
    if 'pbMRQ' in df.columns:
        pb = df['pbMRQ'].fillna(3).values
        pb_clean = np.clip(pb, 0.1, 50)
        factors['f_pb_inv'] = 1.0 / pb_clean
    
    # 6.2 PE倒数（盈利收益率）
    if 'peTTM' in df.columns:
        pe = df['peTTM'].fillna(50).values
        pe_clean = np.where(pe > 0, pe, 50)
        pe_clean = np.clip(pe_clean, 1, 200)
        factors['f_ey'] = 1.0 / pe_clean
    
    # ============================================================
    # 7. 成交量/流动性因子
    # ============================================================
    
    # 7.1 量比
    if 'volume_ratio' in df.columns:
        vr = df['volume_ratio'].fillna(1).values
        factors['f_volume_ratio'] = np.where(
            (vr > 1) & (vr < 3),
            (vr - 1) / 2,
            0
        )
    
    # 7.2 换手率变化
    if 'turn' in df.columns:
        turn = pd.to_numeric(df['turn'], errors='coerce').fillna(0).values
        turn_change = df.groupby('symbol')['turn'].transform(
            lambda x: x / x.rolling(20, min_periods=5).mean() - 1
        ).fillna(0).values
        factors['f_turn_change'] = np.clip(turn_change, -0.5, 2.0)
    
    # ============================================================
    # 8. MACD动量因子
    # ============================================================
    
    if 'macd_hist' in df.columns:
        macd_hist = df['macd_hist'].fillna(0).values
        close_safe = np.where(close > 0, close, 1)
        macd_norm = macd_hist / close_safe
        macd_prev = df.groupby('symbol')['macd_hist'].shift(1).fillna(0).values
        factors['f_macd_momentum'] = np.where(
            (macd_hist > 0) & (macd_hist > macd_prev),
            macd_norm * 100,
            0
        )
        factors['f_macd_momentum'] = np.clip(factors['f_macd_momentum'], 0, 1)
    
    # ============================================================
    # 将因子合并到DataFrame
    # ============================================================
    
    factor_names = []
    for name, values in factors.items():
        df[name] = values
        factor_names.append(name)
    
    logger.info(f"共生成 {len(factor_names)} 个优化因子")
    return df, factor_names


def standardize_factors(df, factor_names):
    """
    对所有因子进行截面标准化（稳健z-score）。
    """
    logger.info(f"正在标准化 {len(factor_names)} 个因子...")
    
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 因子平滑：5日滚动均值
    logger.debug("因子平滑...")
    for col in factor_names:
        df[col] = df.groupby('symbol')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    
    # 截面标准化（稳健z-score）
    logger.debug("截面标准化...")
    
    def robust_zscore(group):
        """稳健z-score：使用median和mad替代mean和std。"""
        median = group.median()
        mad = np.abs(group - median).median()
        mad_std = mad * 1.4826
        mad_std = np.where(mad_std == 0, 1, mad_std)
        result = (group - median) / mad_std
        result = np.clip(result, -5, 5)
        return result
    
    df[factor_names] = df.groupby('date')[factor_names].transform(robust_zscore)
    df[factor_names] = df[factor_names].fillna(0)
    
    logger.info("标准化完成")
    return df
