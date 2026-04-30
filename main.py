# -*- coding: utf-8 -*-
"""
============================================================
  基于 pybroker 的 A股 多因子选股策略（纯技术指标轮动）
  数据来源：baostock（行情）+ akshare（指数成分股）
  指标计算：ta-lib（C底层，162个指标）
  回测框架：pybroker 1.2
  可视化：matplotlib
============================================================

策略流程：
  1. 通过 akshare 获取中证500指数成分股列表
  2. 通过 baostock 获取每只股票的日K线数据
  3. 数据缓存到本地 SQLite 数据库，避免重复下载
  4. 用 ta-lib 计算技术指标（RSI/MACD/KDJ/布林带/ATR/动量/量比）
  5. 7因子截面标准化 + 加权打分，每日选取排名前5的股票
  6. 使用 pybroker（官方轮动交易模式）进行回测
  7. 可视化回测结果（收益曲线/回撤/月度收益/统计面板）

因子体系（7因子纯技术指标）：
  动量 20%、量比 15%、RSI 10%、MACD 15%、KDJ 15%、布林带 15%、ATR 10%

使用方式：
  在你的 Anaconda 环境中直接运行此文件即可：
  python E:/project/pybroker/量化多因子选股策略.py
  python E:/project/pybroker/量化多因子选股策略.py --start 2023-01-01 --end 2025-12-31
"""


# ============================================================
# 模块1：环境配置与库导入
# ============================================================
# 所有必需的第三方库导入，确保 Anaconda 环境已安装这些库

import pybroker as pyb
from pybroker import ExecContext, StrategyConfig, Strategy
from pybroker.common import FeeInfo
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date as date_type
import sqlite3
import os
import time
import socket
import threading
import warnings
import sys
import argparse

# 导入数据源库
import baostock as bs           # 免费A股历史行情数据
import akshare as ak            # A股指数成分股、各类金融数据
import talib                    # 技术指标库（C底层, 162个指标）

# 正常显示中文和负号（Windows下常见问题）
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# 设置 pybroker 数据源缓存（使用 diskcache，类似 SQLite）
pyb.enable_data_source_cache('akshare')
pyb.enable_data_source_cache('custom')

# 初始化日志系统
import logging
from logger_config import setup_logger
logger = setup_logger('strategy', level=logging.INFO)

logger.info("=" * 60)
logger.info("  量化多因子选股策略 启动")
logger.info(f"  pybroker 版本: {pyb.__version__}")
logger.info(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 60)


# ============================================================
# 模块2：全局配置参数
# ============================================================

from config import Config
from data_sources import get_multi_index_stocks, download_all_stocks_data, load_kline_from_sqlite
# from data_sources import get_stock_industry  # 暂时注释，待实现行业中性化
from indicators import compute_all_indicators
from factors import compute_factor_scores
from factor_analysis import evaluate_factors
from strategy import a_share_fee
from backtest import run_backtest, display_results, plot_results, save_results
from data_validation import validate_ohlcv, clean_data, validate_factors

def parse_args():
    """
    解析命令行参数。

    用户只需指定回测区间（--start / --end），
    程序自动推导数据区间 = 回测起始 - DATA_LEAD_DAYS ~ 回测结束。

    用法示例：
      python 量化多因子选股策略.py
      python 量化多因子选股策略.py --start 2023-01-01 --end 2025-12-31
      python 量化多因子选股策略.py --stocks 30 --top-n 10 --cash 500000
      python 量化多因子选股策略.py --start 2023-01-01 --end 2025-12-31 --use-rolling-ml
    """
    default_backtest_end = Config.BACKTEST_END
    default_backtest_start = pd.Timestamp(default_backtest_end) - timedelta(days=365 * 2)
    default_backtest_start = default_backtest_start.strftime('%Y-%m-%d')

    p = argparse.ArgumentParser(
        description='量化多因子选股策略（基于 pybroker + baostock + akshare）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 量化多因子选股策略.py
  python 量化多因子选股策略.py --start 2023-01-01 --end 2025-12-31
  python 量化多因子选股策略.py --stocks 30 --top-n 10 --cash 500000
  python 量化多因子选股策略.py --start 2023-01-01 --end 2025-12-31 --use-rolling-ml

  快速测试模式:
    python main.py --quick-test   # 使用100只股票，1年数据，纯线性加权
    python main.py --fast         # 跳过滚动IC/ML，使用简单线性加权
        """
    )
    p.add_argument('--start', type=str, default=default_backtest_start,
                   help=f'回测起始日期 (默认: {default_backtest_start})')
    p.add_argument('--end', type=str, default=default_backtest_end,
                   help=f'回测结束日期 (默认: {default_backtest_end})')
    p.add_argument('--stocks', type=int, default=0,
                   help='股票数量上限 (0=全部约1800只, 默认: 0)')
    p.add_argument('--top-n', type=int, default=Config.TOP_N_STOCKS,
                   help=f'每日持仓数 (默认: {Config.TOP_N_STOCKS})')
    p.add_argument('--cash', type=int, default=Config.INITIAL_CASH,
                   help=f'初始资金/元 (默认: {Config.INITIAL_CASH})')
    p.add_argument('--use-rolling-ic', action='store_true',
                   help='启用滚动IC加权（无未来函数，推荐）')
    p.add_argument('--use-rolling-ml', action='store_true',
                   help='启用滚动XGBoost机器学习（无未来函数）')
    p.add_argument('--no-ml', action='store_true',
                   help='禁用ML，使用纯线性加权')
    p.add_argument('--fast', action='store_true',
                   help='快速模式：跳过滚动IC/ML，使用简单线性加权')
    p.add_argument('--quick-test', action='store_true',
                   help='快速测试：使用100只股票，1年数据，纯线性加权')
    return p.parse_args()


def main():
    """
    主流程：数据获取 → 指标计算 → 因子打分 → 回测 → 可视化。

    命令行参数（推荐只用这两个）：
      --start  回测起始日期（数据自动向前拉 DATA_LEAD_DAYS 天）
      --end    回测结束日期
      --stocks 股票数量上限（快速测试用）
      --top-n  每日持仓数
      --cash   初始资金
    """
    args = parse_args()

    # ---- 快速测试模式 ----
    if args.quick_test:
        logger.info("🚀 快速测试模式：使用100只股票，1年数据")
        args.stocks = 100
        args.start = '2024-01-01'
        args.end = '2025-01-01'
        # 注意：不设置 args.no_ml，保留 --use-rolling-ic 功能

    # ---- 日期推导 ----
    # 用户指定的是回测区间，数据区间自动向前扩展 DATA_LEAD_DAYS 天
    Config.BACKTEST_START = args.start
    Config.BACKTEST_END = args.end
    Config.DATA_START_DATE = (
        pd.Timestamp(args.start) - timedelta(days=Config.DATA_LEAD_DAYS)
    ).strftime('%Y-%m-%d')
    Config.DATA_END_DATE = args.end
    
    # ---- 其他参数 ----
    Config.STOCK_LIMIT = None if args.stocks == 0 else args.stocks
    Config.TOP_N_STOCKS = args.top_n
    Config.INITIAL_CASH = args.cash
    
    # 快速模式：强制禁用滚动IC/ML
    if args.fast:
        logger.info("⚡ 快速模式：跳过滚动IC/ML，使用简单线性加权")
        use_rolling_ic = False
        use_rolling_ml = False
    else:
        use_rolling_ic = args.use_rolling_ic and not args.no_ml
        use_rolling_ml = args.use_rolling_ml and not args.no_ml and not args.use_rolling_ic

    # 验证日期格式
    for date_str, name in [
        (Config.DATA_START_DATE, '数据起始'),
        (Config.DATA_END_DATE, '数据结束'),
        (Config.BACKTEST_START, '回测起始'),
        (Config.BACKTEST_END, '回测结束'),
    ]:
        try:
            pd.Timestamp(date_str)
        except Exception:
            logger.error(f"{name}日期格式错误: {date_str}")
            return

    logger.info(f"回测区间: {Config.BACKTEST_START} ~ {Config.BACKTEST_END}")
    logger.info(f"数据区间: {Config.DATA_START_DATE} ~ {Config.DATA_END_DATE}")
    logger.info(f"(数据比回测早 {Config.DATA_LEAD_DAYS} 天，用于指标预热)")

    logger.info("=" * 60)
    logger.info("步骤1: 获取指数成分股（沪深300+中证500+中证1000）")
    logger.info("=" * 60)
    stocks_df = get_multi_index_stocks()

    if stocks_df is None or stocks_df.empty:
        logger.error("无法获取成分股列表，回测终止")
        logger.error("提示：akshare 可能暂时不可用，稍后重试或检查网络连接")
        return

    logger.info("=" * 60)
    logger.info("步骤2: 下载/加载日K线数据")
    logger.info("=" * 60)
    # 先尝试下载数据（如有缓存则自动跳过）
    download_all_stocks_data(stocks_df)
    # 从 SQLite 加载数据到 DataFrame
    df = load_kline_from_sqlite(stocks_df)

    if df.empty:
        logger.error("没有数据可供回测，程序退出")
        return

    # 数据校验与清洗
    logger.info("=" * 60)
    logger.info("步骤2.5: 数据校验与清洗")
    logger.info("=" * 60)
    try:
        validate_ohlcv(df, strict=False)
        df = clean_data(df)
    except Exception as e:
        logger.error(f"数据校验失败: {e}")
        return

    logger.info("=" * 60)
    logger.info("步骤3: 计算技术指标")
    logger.info("=" * 60)
    df = compute_all_indicators(df)

    # ---- 获取行业信息并合并 ----
    # 注意：当前策略未使用行业中性化，注释掉以节省时间
    # logger.info("=" * 60)
    # logger.info("步骤3.5: 获取行业信息")
    # logger.info("=" * 60)
    # industry_map = get_stock_industry(stocks_df)
    # if industry_map:
    #     df['industry'] = df['symbol'].map(
    #         lambda x: industry_map.get(x.replace('sh.', '').replace('sz.', '').replace('bj.', ''), '未知')
    #     )
    #     logger.info("行业信息已合并到数据")
    # else:
    #     logger.warning("未获取到行业信息，跳过行业中性化")

    logger.info("=" * 60)
    logger.info("步骤4: 多因子打分与选股")
    logger.info("=" * 60)

    df = compute_factor_scores(df, use_rolling_ml=use_rolling_ml, use_rolling_ic=use_rolling_ic)

    logger.info("=" * 60)
    logger.info("步骤4.5: 因子评估")
    logger.info("=" * 60)
    evaluate_factors(df)

    logger.info("=" * 60)
    logger.info("步骤5: 执行回测")
    logger.info("=" * 60)
    result = run_backtest(df)

    logger.info("=" * 60)
    logger.info("步骤6: 展示结果")
    logger.info("=" * 60)
    display_results(result, df)

    logger.info("=" * 60)
    logger.info("步骤7: 可视化图表")
    logger.info("=" * 60)
    plot_results(result, df)

    logger.info("=" * 60)
    logger.info("步骤8: 保存回测结果")
    logger.info("=" * 60)
    save_results(result, df)

    logger.info("=" * 60)
    logger.info("策略运行完毕！")
    logger.info(f"数据库文件: {Config.SQLITE_DB_PATH}")
    logger.info(f"图表目录: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '图表')}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
