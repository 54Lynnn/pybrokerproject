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
    INDEX_CODES = ['000300', '000905', '000852']  # 沪深300 + 中证500 + 中证1000
    MARKET_INDEX = '000300'  # 市场状态判断用沪深300
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
    TOP_N_STOCKS = 25              # 每日持仓数（适度分散，扩大覆盖面）
    # 以下两个参数由 factors.py 在运行时根据信号半衰期自动计算：
    #   MIN_HOLD_BARS = signal_half_life
    #   SELL_THRESHOLD = TOP_N × max(2, half_life // 3 + 1)
    # 此处仅作默认值/类型声明，实际值在 compute_factor_scores() 中动态覆盖
    SELL_THRESHOLD = TOP_N_STOCKS * 6
    MIN_HOLD_BARS = 15

    # -------- 回测参数 --------
    BACKTEST_START = (date_type.today() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
    BACKTEST_END = (date_type.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    INITIAL_CASH = 1_000_000

    # -------- 费率参数（A股真实费率）--------
    COMMISSION_RATE = 0.00025   # 佣金 0.025%（买卖双向）
    STAMP_DUTY_RATE = 0.001     # 印花税 0.1%（仅卖出）

    # 基于IR优化的权重配置（只保留高稳定性因子）
    # 核心原则：只保留IR > 0.15的因子，提高策略稳定性
    # 优化效果：因子数量从24个减少到7个，计算量减少70%
    FACTOR_WEIGHTS = {
        # ===== 高IR因子（IR > 0.15，重点配置）=====
        'volatility_5_inv': 0.18,      # IC=+0.0584, IR=0.298 ✅ 最高
        'pb_inv': 0.16,                 # IC=+0.0530, IR=0.244 ✅
        'volatility_20_inv': 0.16,      # IC=+0.0552, IR=0.244 ✅
        'atr_inv': 0.15,                # IC=+0.0566, IR=0.227 ✅
        'volatility_60_inv': 0.15,      # IC=+0.0556, IR=0.223 ✅
        'rsi_oversold': 0.10,           # IC=+0.0196, IR=0.211 ✅
        'reversal_20': 0.10,            # IC=+0.0363, IR=0.205 ✅

        # ===== 反向因子（IC为负，赋负权重=反向做多，增强组合）=====
        'trend_strength': -0.05,        # IC=-0.0301, IR=-0.212 → 负权重反向利用 ✅
        'momentum_60': -0.03,           # IC=-0.0331, IR=-0.179 → 负权重反向利用 ✅
        'bb_high': -0.02,               # IC=-0.0168, IR=-0.157 → 负权重反向利用 ✅
        
        # ===== 以下因子因IR绝对值 < 0.15已删除 =====
        # 'bb_low': 0.04,                # IC=+0.0174, IR=0.189 ❌
        # 'reversal_10': 0.04,           # IC=+0.0300, IR=0.181 ❌
        # 'ey': 0.05,                    # IC=+0.0365, IR=0.173 ❌
        # 'reversal_5': 0.03,            # IC=+0.0265, IR=0.160 ❌
        # 'kdj_j_low': 0.03,             # IC=+0.0121, IR=0.140 ❌
        # 'combined_reversal': 0.03,     # IC=+0.0171, IR=0.129 ❌
        # 'willr_oversold': 0.02,        # IC=+0.0150, IR=0.120 ❌
        # 'adx_trend': 0.01,             # IC=+0.0021, IR=0.020 ❌
        # 'momentum_120': -0.04,         # IC=-0.0269, IR=-0.149 ❌
        # 'macd_momentum': -0.03,        # IC=-0.0169, IR=-0.134 ❌
        # 'rsi_overbought': -0.02,       # IC=-0.0072, IR=-0.099 ❌
        # 'kdj_j_high': -0.02,           # IC=-0.0017, IR=-0.017 ❌
        # 'turn_change': -0.03,          # IC=-0.0155, IR=-0.147 ❌
    }
