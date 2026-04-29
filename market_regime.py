# -*- coding: utf-8 -*-
"""
市场状态判断与动态权重调整 — 专业优化版

优化要点：
  1. 增加市场状态判断的稳定性
  2. 优化各状态下的因子权重配置
  3. 增加趋势跟踪权重
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_market_regime(df_index):
    """
    根据中证500指数数据判断市场状态。
    
    判断逻辑（优化版）：
      1. 计算指数的20日、60日均线
      2. 计算20日波动率
      3. 计算近期涨跌幅（20日、60日）
      4. 综合判断市场状态：
         - 强牛市: 价格>MA20>MA60，20日涨幅>10%，波动率适中
         - 牛市: 价格>MA20>MA60
         - 强熊市: 价格<MA20<MA60，20日跌幅>10%
         - 熊市: 价格<MA20<MA60
         - 震荡: 均线纠缠，波动率低
         - 高波: 波动率极高
    """
    if df_index is None or df_index.empty:
        logger.warning("指数数据为空，无法判断市场状态")
        return None
    
    df = df_index.copy().sort_values('date').reset_index(drop=True)
    
    # 计算均线
    df['ma_20'] = df['close'].rolling(window=20, min_periods=10).mean()
    df['ma_60'] = df['close'].rolling(window=60, min_periods=30).mean()
    
    # 计算收益率和波动率
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20, min_periods=10).std() * np.sqrt(252)
    
    # 计算近期涨跌幅
    df['ret_20d'] = df['close'] / df['close'].shift(20) - 1
    df['ret_60d'] = df['close'] / df['close'].shift(60) - 1
    
    def classify_regime(row):
        """单条记录分类（优化版）。"""
        close = row['close']
        ma20 = row['ma_20']
        ma60 = row['ma_60']
        vol = row['volatility_20']
        ret_20d = row['ret_20d']
        
        # 数据不足时默认中性
        if pd.isna(ma60):
            return 'neutral'
        
        # 高波动状态（年化波动率 > 35%）
        if vol > 0.35:
            return 'high_vol'
        
        # 强牛市：均线多头排列 + 近期大涨
        if close > ma20 and ma20 > ma60 and ret_20d > 0.10:
            return 'strong_bull'
        
        # 牛市：价格在MA20上，MA20在MA60上
        if close > ma20 and ma20 > ma60:
            return 'bull'
        
        # 强熊市：均线空头排列 + 近期大跌
        if close < ma20 and ma20 < ma60 and ret_20d < -0.10:
            return 'strong_bear'
        
        # 熊市：价格在MA20下，MA20在MA60下
        if close < ma20 and ma20 < ma60:
            return 'bear'
        
        # 震荡：均线纠缠，波动率低
        if abs(ma20 / ma60 - 1) < 0.05 and vol < 0.25:
            return 'neutral'
        
        # 默认根据价格位置判断
        if close > ma20:
            return 'bull_weak'
        else:
            return 'bear_weak'
    
    df['market_regime'] = df.apply(classify_regime, axis=1)
    
    # 统计各状态占比
    regime_counts = df['market_regime'].value_counts()
    logger.info("市场状态分布:")
    for regime, count in regime_counts.items():
        logger.info(f"  {regime}: {count} 天 ({count/len(df)*100:.1f}%)")
    
    return df[['date', 'market_regime', 'ma_20', 'ma_60', 'volatility_20']]


def get_dynamic_weights(market_regime, base_weights):
    """
    根据市场状态返回动态调整的因子权重（专业优化版）。
    
    调整逻辑：
      - 强牛市: 最大化动量权重，降低反转
      - 牛市: 加大动量、趋势因子
      - 熊市: 加大低波、反转、估值因子
      - 强熊市: 最大化低波和反转
      - 震荡: 均衡配置，加大均值回复
      - 高波: 加大低波，降低动量
    """
    weights = base_weights.copy()
    
    # 定义各状态下的权重调整（适配修复后的因子）
    regime_adjustments = {
        'strong_bull': {
            'boost': ['momentum_60', 'momentum_120', 'trend_strength', 'adx_trend'],
            'reduce': ['reversal_5', 'reversal_10', 'combined_reversal', 'rsi_oversold'],
            'boost_factor': 1.5,
            'reduce_factor': 0.3
        },
        'bull': {
            'boost': ['momentum_60', 'trend_strength', 'volume_ratio'],
            'reduce': ['reversal_5', 'combined_reversal'],
            'boost_factor': 1.3,
            'reduce_factor': 0.5
        },
        'bull_weak': {
            'boost': ['momentum_60', 'trend_strength', 'pb_inv'],
            'reduce': ['reversal_5'],
            'boost_factor': 1.1,
            'reduce_factor': 0.7
        },
        'bear': {
            'boost': ['reversal_5', 'reversal_10', 'reversal_20', 'volatility_20_inv', 'volatility_60_inv', 'atr_inv', 'combined_reversal', 'rsi_oversold'],
            'reduce': ['momentum_60', 'momentum_120'],
            'boost_factor': 1.3,
            'reduce_factor': 0.5
        },
        'strong_bear': {
            'boost': ['reversal_5', 'reversal_10', 'reversal_20', 'volatility_20_inv', 'volatility_60_inv', 'atr_inv', 'combined_reversal', 'rsi_oversold', 'willr_oversold'],
            'reduce': ['momentum_60', 'momentum_120', 'trend_strength'],
            'boost_factor': 1.5,
            'reduce_factor': 0.3
        },
        'bear_weak': {
            'boost': ['reversal_20', 'volatility_20_inv', 'pb_inv', 'ey'],
            'reduce': ['momentum_60'],
            'boost_factor': 1.1,
            'reduce_factor': 0.7
        },
        'neutral': {
            'boost': ['reversal_10', 'reversal_20', 'combined_reversal', 'bb_low', 'macd_momentum'],
            'reduce': ['momentum_60'],
            'boost_factor': 1.2,
            'reduce_factor': 0.6
        },
        'high_vol': {
            'boost': ['volatility_20_inv', 'volatility_60_inv', 'atr_inv', 'pb_inv'],
            'reduce': ['momentum_60', 'momentum_120'],
            'boost_factor': 1.4,
            'reduce_factor': 0.4
        }
    }
    
    # 应用调整
    if market_regime in regime_adjustments:
        adj = regime_adjustments[market_regime]
        
        for factor in adj['boost']:
            key = 'f_' + factor
            if key in weights:
                weights[key] *= adj['boost_factor']
        
        for factor in adj['reduce']:
            key = 'f_' + factor
            if key in weights:
                weights[key] *= adj['reduce_factor']
    
    # 归一化，确保权重和为1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    return weights
