# -*- coding: utf-8 -*-
"""全局配置参数"""

from datetime import datetime, timedelta, date as date_type
import os

class Config:
    """策略全局配置类，集中管理所有参数"""

    # -------- 数据参数 --------
    # 默认结束日期 = 昨天（最新交易日）
    _today = date_type.today()
    _yesterday = (_today - timedelta(days=1)).strftime('%Y-%m-%d')
    _two_years_ago = (_today - timedelta(days=365 * 2)).strftime('%Y-%m-%d')

    DATA_START_DATE = _two_years_ago     # 数据起始（最近两年）
    DATA_END_DATE = _yesterday           # 数据结束（昨天）
    DATA_LEAD_DAYS = 90                  # 数据领先回测天数（日历日）
    # 说明：指标计算需要历史数据预热（20日均线20天、RSI收敛约30天等），
    # 数据起始必须比回测起始早至少 DATA_LEAD_DAYS 天（90日历日≈60个交易日）。
    SQLITE_DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'stock_kline_cache.db'
    )

    # -------- 成分股参数 --------
    INDEX_CODE = '000905'               # 中证500指数代码
    STOCK_LIMIT = None                  # 股票数量限制（None=全部，设为50可快速测试）

    # -------- 因子参数 --------
    RSI_PERIOD = 14                     # RSI 计算周期
    MACD_FAST = 12                      # MACD 快线周期
    MACD_SLOW = 26                      # MACD 慢线周期
    MACD_SIGNAL = 9                     # MACD 信号线周期
    KDJ_PERIOD = 9                      # KDJ 计算周期
    MOMENTUM_PERIOD = 20                # 动量计算周期
    VOLUME_MA_PERIOD = 20               # 成交量均线周期
    BB_PERIOD = 20                      # 布林带周期
    ATR_PERIOD = 14                     # ATR 周期

    # -------- 选股参数 --------
    TOP_N_STOCKS = 10                  # 每日持仓股票数量上限
    SELL_THRESHOLD = TOP_N_STOCKS * 3  # 卖出阈值：TOP_N×3（半衰期~8天）
    MIN_HOLD_BARS = 8                  # 最低持有天数（信号半衰期≈8天）
    FACTOR_WEIGHTS = {                 # 9因子权重（总和=1）
        'momentum_20d': 0.12,          # 20日动量(反转)
        'momentum_5d': 0.08,           # 5日动量(追涨)
        'volume_ratio': 0.05,          # 量比(反转)
        'rsi_score': 0.15,            # RSI
        'macd_score': 0.08,           # MACD(反转)
        'kdj_score': 0.12,            # KDJ
        'bb_score': 0.12,             # 布林带
        'cci_score': 0.08,            # CCI
        'atr_score': 0.20,            # ATR（最强）
    }

    # -------- 资金与风控参数 --------
    INITIAL_CASH = 1_000_000            # 初始资金（元）
    COMMISSION_RATE = 0.00025           # 佣金费率（买卖双向）：0.025%
    STAMP_DUTY_RATE = 0.001             # 印花税率（卖出单向）：0.1%
    STOP_LOSS_PCT = -0.10               # 止损线：亏损10%止损
    TAKE_PROFIT_PCT = 0.30              # 止盈线：盈利30%止盈
    A_SHARE_LOT = 100                   # A股每手100股

    # -------- 回测参数 --------
    # BACKTEST_START 默认值在 main() 中根据实际 DATA_START_DATE 动态计算
    BACKTEST_START = _yesterday           # 占位，main() 中会覆写
    BACKTEST_END = _yesterday             # 回测结束（昨天）
