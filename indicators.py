# -*- coding: utf-8 -*-
"""技术指标计算 (ta-lib) — 向量化优化版"""

from config import Config
import numpy as np
import pandas as pd
import talib
import logging

logger = logging.getLogger(__name__)


def _ta_lib_by_group(df, func, *col_names, **kwargs):
    """
    对分组数据应用ta-lib函数，返回展平的Series。
    
    这是向量化计算的核心辅助函数：
    1. 按symbol分组
    2. 对每个组的指定列应用ta-lib函数
    3. 将结果展平后返回
    """
    def apply_func(group):
        arrays = [group[col].values.astype(float) for col in col_names]
        return pd.Series(func(*arrays, **kwargs), index=group.index)
    
    result = df.groupby('symbol', group_keys=False).apply(apply_func, include_groups=False)
    return result.values


def compute_all_indicators(df):
    """
    对所有股票的 DataFrame 向量化计算技术指标。
    
    优化策略：
      1. 使用 groupby().transform() 进行向量化计算，避免Python级循环
      2. 对ta-lib函数使用分组apply，减少函数调用开销
      3. 保持与原版完全相同的输出列名和计算逻辑
    """
    logger.info("正在计算技术指标（向量化模式）...")
    df = df.copy().sort_values(['symbol', 'date']).reset_index(drop=True)
    
    n_symbols = df['symbol'].nunique()
    n_rows = len(df)
    logger.info(f"数据规模: {n_symbols} 只股票, {n_rows} 条记录")

    # ---- 准备基础数据列 ----
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)

    # ---- RSI(14) ----
    logger.debug("计算 RSI...")
    df['rsi'] = _ta_lib_by_group(df, talib.RSI, 'close', timeperiod=Config.RSI_PERIOD)

    # ---- MACD(12/26/9) ----
    logger.debug("计算 MACD...")
    macd_dif = _ta_lib_by_group(df, lambda c, **kw: talib.MACD(c, **kw)[0], 'close',
                                 fastperiod=Config.MACD_FAST, slowperiod=Config.MACD_SLOW,
                                 signalperiod=Config.MACD_SIGNAL)
    macd_dea = _ta_lib_by_group(df, lambda c, **kw: talib.MACD(c, **kw)[1], 'close',
                                 fastperiod=Config.MACD_FAST, slowperiod=Config.MACD_SLOW,
                                 signalperiod=Config.MACD_SIGNAL)
    macd_hist = _ta_lib_by_group(df, lambda c, **kw: talib.MACD(c, **kw)[2], 'close',
                                  fastperiod=Config.MACD_FAST, slowperiod=Config.MACD_SLOW,
                                  signalperiod=Config.MACD_SIGNAL)
    df['macd_dif'] = macd_dif
    df['macd_dea'] = macd_dea
    df['macd_hist'] = macd_hist

    # ---- KDJ(9) ----
    logger.debug("计算 KDJ...")
    k = _ta_lib_by_group(df, lambda h, l, c, **kw: talib.STOCH(h, l, c, **kw)[0],
                          'high', 'low', 'close',
                          fastk_period=Config.KDJ_PERIOD, slowk_period=3, slowd_period=3)
    d = _ta_lib_by_group(df, lambda h, l, c, **kw: talib.STOCH(h, l, c, **kw)[1],
                          'high', 'low', 'close',
                          fastk_period=Config.KDJ_PERIOD, slowk_period=3, slowd_period=3)
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = 3 * k - 2 * d

    # ---- 布林带 %b ----
    logger.debug("计算 布林带...")
    bb_upper = _ta_lib_by_group(df, lambda c, **kw: talib.BBANDS(c, **kw)[0], 'close',
                                 timeperiod=Config.BB_PERIOD, nbdevup=2, nbdevdn=2)
    bb_mid = _ta_lib_by_group(df, lambda c, **kw: talib.BBANDS(c, **kw)[1], 'close',
                               timeperiod=Config.BB_PERIOD, nbdevup=2, nbdevdn=2)
    bb_lower = _ta_lib_by_group(df, lambda c, **kw: talib.BBANDS(c, **kw)[2], 'close',
                                 timeperiod=Config.BB_PERIOD, nbdevup=2, nbdevdn=2)
    df['bb_pct_b'] = np.where(bb_upper != bb_lower,
                               (close - bb_lower) / (bb_upper - bb_lower), 0.5)

    # ---- 20日动量 ----
    logger.debug("计算 动量...")
    df['return_20d'] = df.groupby('symbol')['close'].transform(
        lambda x: x.values / np.roll(x.values, Config.MOMENTUM_PERIOD) - 1
    )
    # 修正：np.roll会把末尾数据滚到开头，需要把前20天置为NaN
    df['return_20d'] = df.groupby('symbol')['return_20d'].transform(
        lambda x: x.where(x.index >= x.index[0] + Config.MOMENTUM_PERIOD)
    )

    # ---- 量比 ----
    logger.debug("计算 量比...")
    vol_ma = df.groupby('symbol')['volume'].transform(
        lambda x: pd.Series(x.values).rolling(window=Config.VOLUME_MA_PERIOD, min_periods=1).mean().values
    )
    df['volume_ratio'] = np.divide(volume, vol_ma, out=np.ones_like(volume, dtype=float), where=vol_ma > 0)

    # ---- ATR(14) ----
    logger.debug("计算 ATR...")
    df['atr'] = _ta_lib_by_group(df, talib.ATR, 'high', 'low', 'close', timeperiod=Config.ATR_PERIOD)

    # ---- 均线 ----
    logger.debug("计算 均线...")
    df['ma_5'] = _ta_lib_by_group(df, talib.SMA, 'close', timeperiod=5)
    df['ma_20'] = _ta_lib_by_group(df, talib.SMA, 'close', timeperiod=20)

    # ---- 波动率(20) ----
    logger.debug("计算 波动率...")
    df['volatility'] = df.groupby('symbol')['close'].transform(
        lambda x: pd.Series(x.values).pct_change().rolling(window=20, min_periods=1).std().values
    )

    # ---- ADX(14) ----
    logger.debug("计算 ADX...")
    df['adx'] = _ta_lib_by_group(df, talib.ADX, 'high', 'low', 'close', timeperiod=14)

    # ---- Williams %R(14) ----
    logger.debug("计算 WILLR...")
    df['willr'] = _ta_lib_by_group(df, talib.WILLR, 'high', 'low', 'close', timeperiod=14)

    # ---- OBV ----
    logger.debug("计算 OBV...")
    df['obv'] = _ta_lib_by_group(df, talib.OBV, 'close', 'volume')

    # ---- CCI(20) ----
    logger.debug("计算 CCI...")
    df['cci'] = _ta_lib_by_group(df, talib.CCI, 'high', 'low', 'close', timeperiod=20)

    # ---- MFI(14) ----
    logger.debug("计算 MFI...")
    df['mfi'] = _ta_lib_by_group(df, talib.MFI, 'high', 'low', 'close', 'volume', timeperiod=14)

    logger.info("指标计算完成：RSI / MACD / KDJ / 布林带 / ATR / 动量 / 量比 / 均线 / 波动率 / ADX / WILLR / OBV / CCI / MFI")
    return df
