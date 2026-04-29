# -*- coding: utf-8 -*-
"""pybroker 交易策略 (轮动模式: before_exec + execution)"""

from config import Config
from decimal import Decimal
from pybroker import ExecContext, StrategyConfig, Strategy
from pybroker.common import FeeInfo
import numpy as np
import pandas as pd
import pybroker as pyb
import factors

def rank(ctxs: dict[str, ExecContext]):
    """
    before_exec —— 每个 bar 开始前，在所有活跃股票中找出今日选股。

    官方轮动模式：在 before_exec 中完成跨股票的排名/筛选，
    结果存入全局参数 pyb.param，供 exec 函数读取。

    逻辑：根据当前 bar 日期，从预计算的 factors.SELECTION_MAP 中
    查找今日应持有的股票列表

    Args:
        ctxs: {symbol: ExecContext} 当前活跃的所有股票上下文
    """
    if not ctxs:
        return

    first_ctx = next(iter(ctxs.values()))
    bar_date = pd.Timestamp(first_ctx.dt)

    # 信号是 T 日收盘产生 → T+1 日执行 → 找前一天
    target_date = bar_date - pd.Timedelta(days=1)
    for _ in range(15):   # 15天覆盖最长假期（春节7天+两端周末）
        if target_date in factors.SELECTION_MAP:
            break
        target_date -= pd.Timedelta(days=1)

    if target_date in factors.SELECTION_MAP:
        all_selected = list(factors.SELECTION_MAP[target_date])
        # top_symbols = 前 TOP_N 只（买入候选）
        # keep_symbols = 全部前 SELL_THRESHOLD 只（持仓股票排名在前 SELL_THRESHOLD 就保留）
        top_list = all_selected[:Config.TOP_N_STOCKS]
        keep_list = all_selected
    else:
        top_list = []
        keep_list = []

    pyb.param('top_symbols', top_list)
    pyb.param('keep_symbols', keep_list)
    pyb.param('target_size', 1.0 / Config.TOP_N_STOCKS)


def execute_strategy(ctx: ExecContext):
    """
    per-symbol 执行函数 —— 根据 before_exec 排出的 top_symbols 下单。

    官方轮动模式中的 rotate 函数：
      - 如果持仓且不在 top_symbols 中 → 全部卖出
      - 如果无持仓且在 top_symbols 中 → 按目标仓位买入

    新增止损机制：
      - 个股止损：-8%
      - 动态止盈：盈利20%后回撤10%卖出

    关键 API（来自官方文档）：
      - ctx.long_pos() → Position 或 None
      - ctx.close[-1]  → 最新收盘价
      - ctx.sell_all_shares() → 清仓
      - ctx.calc_target_shares(target) → 计算目标持仓股数
      - ctx.buy_shares / ctx.buy_limit_price → 下单
    """
    top_symbols = pyb.param('top_symbols')
    keep_symbols = pyb.param('keep_symbols')
    target_size = pyb.param('target_size')

    pos = ctx.long_pos()

    # 情况1：有持仓 → 检查止损/止盈条件
    if pos:
        # 计算当前盈亏比例（使用持仓平均成本）
        current_price = ctx.close[-1]

        # 从 pos.entries 计算加权平均入场价格
        entry_price = 0.0
        total_shares = 0
        for entry in pos.entries:
            entry_price += float(entry.price) * float(entry.shares)
            total_shares += float(entry.shares)
        if total_shares > 0:
            entry_price = entry_price / total_shares

        if entry_price > 0:
            pnl_pct = (current_price - entry_price) / entry_price

            # 止损：亏损超过8%
            if pnl_pct < -0.08:
                ctx.sell_all_shares()
                return

            # 动态止盈：盈利超过20%后，回撤10%卖出
            if pnl_pct > 0.20:
                # 使用pybroker的param记录每只股票的最高价
                # 注意：param是全局的，必须用股票代码作为key前缀隔离
                highest_price_key = f'highest_price_{ctx.symbol}'
                highest_price = pyb.param(highest_price_key)
                
                # 首次触发止盈跟踪：初始化最高价为当前价（而非entry_price）
                # 因为entry_price到current_price之间可能已经涨了很多
                if highest_price is None:
                    highest_price = current_price
                    pyb.param(highest_price_key, highest_price)
                
                # 更新最高价
                if current_price > highest_price:
                    highest_price = current_price
                    pyb.param(highest_price_key, highest_price)
                
                # 从最高点回撤10%触发止盈
                if current_price < highest_price * 0.90:
                    ctx.sell_all_shares()
                    # 清仓后清除该股票的最高价记录，避免内存泄漏
                    pyb.param(highest_price_key, None)
                    return

        # 轮动卖出：排名在 SELL_THRESHOLD 以外则清仓
        if ctx.symbol not in keep_symbols:
            ctx.sell_all_shares()
        return

    # 情况2：无持仓且在前 TOP_N → 买入（需满足趋势过滤）
    if ctx.symbol in top_symbols:
        # 趋势过滤：仅买入收盘价在 20 日均线之上的股票（避免接飞刀）
        if ctx.bars >= 20 and ctx.close[-1] >= np.mean(ctx.close[-20:]):
            ctx.buy_shares = ctx.calc_target_shares(target_size)
            ctx.buy_limit_price = ctx.close[-1]
            ctx.hold_bars = Config.MIN_HOLD_BARS  # 强制持有至少 N 天


# ============================================================
# 模块7：回测执行
# ============================================================

def a_share_fee(info: FeeInfo) -> Decimal:
    """
    A 股真实费率函数。

    收费标准：
      - 佣金 0.025%（买卖双向）
      - 印花税 0.1%（仅卖出）

    Args:
        info: pybroker 传入的 FeeInfo，含 order_type / shares / fill_price

    Returns:
        Decimal: 该笔交易的费用（元）
    """
    amount = info.shares * info.fill_price
    commission = float(amount) * Config.COMMISSION_RATE
    if info.order_type == 'sell':
        stamp = float(amount) * Config.STAMP_DUTY_RATE
    else:
        stamp = 0.0
    return Decimal(str(commission + stamp))
