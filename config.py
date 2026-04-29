# -*- coding: utf-8 -*-
"""
全局配置参数 — 基于IC分析优化版

根据2023-2025年回测IC分析调整：
  - 反转因子IC为正但较低（+0.02~+0.03）
  - 动量因子IC为负（-0.02），说明方向搞反了！
  - 趋势因子IC为负（-0.02），说明方向搞反了！
  - 低波因子IC最高（+0.05）
  
修复方向：
  1. 反转因子降低权重（牛市中反转效应弱）
  2. 动量因子改为正向（强者恒强）
  3. 趋势因子改为正向（趋势跟踪）
  4. 低波因子保持高权重（IC最高）
"""

from datetime import datetime, timedelta, date as date_type
import os

class Config:
    """策略全局配置类"""

    # -------- 数据参数 --------
    _today = date_type.today()
    _yesterday = (_today - timedelta(days=1)).strftime('%Y-%m-%d')
    _two_years_ago = (_today - timedelta(days=365 * 2)).strftime('%Y-%m-%d')

    DATA_START_DATE = _two_years_ago
    DATA_END_DATE = _yesterday
    DATA_LEAD_DAYS = 400  # 至少1年+缓冲，确保XGBoost有足够训练数据
    SQLITE_DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'stock_kline_cache.db'
    )

    # -------- 成分股参数 --------
    INDEX_CODE = '000905'
    STOCK_LIMIT = None

    # -------- 因子参数 --------
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    KDJ_PERIOD = 9
    MOMENTUM_PERIOD = 20
    VOLUME_MA_PERIOD = 20
    BB_PERIOD = 20
    ATR_PERIOD = 14

    # -------- 选股参数 --------
    TOP_N_STOCKS = 10
    SELL_THRESHOLD = TOP_N_STOCKS * 3
    MIN_HOLD_BARS = 20

    # 基于IC分析的全新权重配置（修复动量/趋势方向）
    FACTOR_WEIGHTS = {
        # ===== 低波类因子（IC最高，权重25%）=====
        'volatility_20_inv': 0.10,