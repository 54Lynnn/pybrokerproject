# -*- coding: utf-8 -*-
"""技术指标计算 (ta-lib)"""

from config import Config
import numpy as np
import pandas as pd
import talib

def compute_indicators_for_group(group_df):
    """
    用 ta-lib 计算单只股票的全部技术指标。

    指标一览：
      - RSI(14)        — 相对强弱
      - MACD(12/26/9)  — DIF / DEA / 柱
      - KDJ(9)         — K / D / J
      - 布林带(20)      — %b 位置
      - 20日动量        — 近20日涨跌幅
      - 量比            — 成交量 / 20日均量
      - ATR(14)        — 平均真实波幅
      - MA5 / MA20     — 均线
    """
    group_df = group_df.copy().sort_values('date').reset_index(drop=True)
    high = group_df['high'].values.astype(float)
    low = group_df['low'].values.astype(float)
    close = group_df['close'].values.astype(float)
    volume = group_df['volume'].values.astype(float)

    # ---- RSI ----
    group_df['rsi'] = talib.RSI(close, timeperiod=Config.RSI_PERIOD)

    # ---- MACD ----
    macd_dif, macd_dea, macd_hist = talib.MACD(
        close, fastperiod=Config.MACD_FAST, slowperiod=Config.MACD_SLOW,
        signalperiod=Config.MACD_SIGNAL)
    group_df['macd_dif'] = macd_dif
    group_df['macd_dea'] = macd_dea
    group_df['macd_hist'] = macd_hist

    # ---- KDJ ----
    k, d = talib.STOCH(high, low, close,
                       fastk_period=Config.KDJ_PERIOD, slowk_period=3, slowd_period=3)
    j = 3 * k - 2 * d
    group_df['kdj_k'] = k
    group_df['kdj_d'] = d
    group_df['kdj_j'] = j

    # ---- 布林带 %b ----
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=Config.BB_PERIOD, nbdevup=2, nbdevdn=2)
    group_df['bb_pct_b'] = np.where(bb_upper != bb_lower,
                                     (close - bb_lower) / (bb_upper - bb_lower), 0.5)

    # ---- 20日动量 ----
    group_df['return_20d'] = close / pd.Series(close).shift(Config.MOMENTUM_PERIOD).values - 1

    # ---- 量比 ----
    vol_ma = pd.Series(volume).rolling(window=Config.VOLUME_MA_PERIOD, min_periods=1).mean().values
    group_df['volume_ratio'] = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma > 0)

    # ---- ATR ----
    group_df['atr'] = talib.ATR(high, low, close, timeperiod=Config.ATR_PERIOD)

    # ---- 均线（保留用于参考） ----
    group_df['ma_5'] = talib.SMA(close, timeperiod=5)
    group_df['ma_20'] = talib.SMA(close, timeperiod=20)

    return group_df


def compute_all_indicators(df):
    """
    对所有股票的 DataFrame 分组计算技术指标。

    使用手动分组循环代替 groupby.apply，避免不同 pandas 版本
    中 include_groups=False 等参数导致的列丢失问题。
    """
    print(f"\n[因子] 正在计算技术指标...")

    result_list = []
    symbols = df['symbol'].unique()
    total_syms = len(symbols)

    for i, sym in enumerate(symbols):
        group_df = df[df['symbol'] == sym].copy()
        group_df = compute_indicators_for_group(group_df)
        result_list.append(group_df)

        if (i + 1) % 100 == 0 or (i + 1) == total_syms:
            print(f"  指标计算进度: {i+1}/{total_syms} ({100*(i+1)/total_syms:.0f}%)")

    df = pd.concat(result_list, ignore_index=True)
    print(f"  ✓ 指标计算完成：RSI / MACD / KDJ / 布林带 / ATR / 动量 / 量比 / 均线")
    return df
