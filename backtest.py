# -*- coding: utf-8 -*-
"""回测执行、结果展示、可视化、结果持久化"""

from config import Config
from factors import build_daily_selections
import factors
from strategy import rank, execute_strategy, a_share_fee
from datetime import datetime, timedelta
from pybroker import ExecContext, StrategyConfig, Strategy
from pylab import mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pybroker as pyb
import json
import logging

logger = logging.getLogger(__name__)


def _compute_equal_weight_benchmark(df, start_date, end_date):
    """
    计算成分股每日等权平均收益率作为基准。
    
    使用选股池内所有股票（沪深300+中证500+中证1000，约1800只）
    的每日涨跌幅等权平均作为基准收益率。
    
    与市值加权指数不同，等权基准对每只股票一视同仁，
    更公平地反映多因子选股策略的超额收益能力。
    
    Args:
        df: 包含 pctChg 列的完整数据 DataFrame
        start_date, end_date: 回测区间
    
    Returns:
        pd.Series: index=date, values=日收益率（小数，如0.01表示1%）
    """
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    
    # 等权平均：每天所有股票的 pctChg 均值
    benchmark = df_filtered.groupby('date')['pctChg'].mean() / 100.0
    benchmark = benchmark.sort_index()
    
    n_stocks = df_filtered['symbol'].nunique()
    print(f"  ✓ 等权基准：{n_stocks} 只股票，{len(benchmark)} 个交易日")
    return benchmark


def run_backtest(df):
    """
    使用 pybroker 执行多因子选股策略回测（官方轮动交易模式）。

    架构（参考官方 Rotational Trading）：
      1. build_daily_selections → 构建 {date: {symbols}} 选股表
      2. strategy.set_before_exec(rank) → 每日跨股票排名
      3. strategy.add_execution(execute_strategy, symbols) → per-symbol 下单
      4. strategy.backtest(warmup=20) → 执行回测
    """

    logger.info("=" * 60)
    logger.info("开始 pybroker 回测")
    logger.info(f"初始资金: {Config.INITIAL_CASH:,.0f} 元")
    logger.info(f"回测区间: {Config.BACKTEST_START} ~ {Config.BACKTEST_END}")
    logger.info(f"持仓上限: {Config.TOP_N_STOCKS} 只")
    logger.info("=" * 60)

    # ---- 构建每日选股映射表 ----
    logger.info("构建选股信号...")
    factors.SELECTION_MAP = build_daily_selections(df)

    if not factors.SELECTION_MAP:
        logger.error("选股信号为空！")
        return None

    # 初始化全局参数默认值
    pyb.param('top_symbols', [])
    pyb.param('keep_symbols', [])
    pyb.param('target_size', 1.0 / Config.TOP_N_STOCKS)

    # ---- 准备 OHLCV 数据 ----
    # pybroker 只认这些标准列，不含 selected
    ohlcv_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    df_bt = df[ohlcv_cols].copy()

    # 过滤回测时间范围（留前面一些天给 warmup）
    df_bt = df_bt[
        (df_bt['date'] >= Config.BACKTEST_START) &
        (df_bt['date'] <= Config.BACKTEST_END)
    ]

    # 删除 OHLCV 缺失的行
    df_bt = df_bt.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

    # 严格排序 + 重置索引
    df_bt = df_bt.sort_values(['symbol', 'date']).reset_index(drop=True)

    if df_bt.empty:
        logger.error("回测数据为空！")
        return None

    logger.info(f"回测数据: {df_bt['symbol'].nunique()} 只股票, {len(df_bt)} 条记录")

    symbols = df_bt['symbol'].unique().tolist()

    # ---- 创建策略配置 ----
    config = StrategyConfig(
        initial_cash=Config.INITIAL_CASH,
        max_long_positions=Config.TOP_N_STOCKS,  # 最大同时持仓数
        fee_mode=a_share_fee,                     # A股真实费率（买0.025%/卖0.125%）
    )

    # ---- 创建策略对象 ----
    strategy = Strategy(
        df_bt,
        start_date=Config.BACKTEST_START,
        end_date=Config.BACKTEST_END,
        config=config,
    )

    # ---- 设置 before_exec（跨股票排名）----
    strategy.set_before_exec(rank)

    # ---- 绑定执行函数（per-symbol 下单）----
    strategy.add_execution(
        execute_strategy,
        symbols=symbols,
    )

    # ---- 执行回测 ----
    logger.info("正在执行回测，请稍候...")
    print("  ⏳ 回测执行中（约30秒），请稍候...")
    pyb.disable_progress_bar()
    result = strategy.backtest(
        warmup=20,
        disable_parallel=False,
    )
    pyb.enable_progress_bar()
    print("  ✅ 回测完成！")

    logger.info("回测完成！")
    return result


def display_results(result, df):
    """
    打印回测结果的核心指标。

    输出指标包括：
      - 交易次数、胜率
      - 总收益率、最大回撤
      - 夏普比率、索提诺比率
      - 盈亏比（Profit Factor）
      - 基准收益率、超额收益率

    Args:
        result: pybroker.TestResult 对象
        df: 包含回测数据的 DataFrame（用于计算基准收益）
    """
    if result is None:
        print("  ✗ 无回测结果可展示")
        return

    print(f"\n{'=' * 60}")
    print(f"  回测结果")
    print(f"{'=' * 60}")

    # 使用 result.metrics (EvalMetrics dataclass) 直接取字段
    m = result.metrics

    # ---- 计算基准数据（成分股等权平均） ----
    # 用选股池内所有股票（沪深300+中证500+中证1000）的每日等权平均作为基准
    benchmark_data = {}
    benchmark_returns = _compute_equal_weight_benchmark(df, Config.BACKTEST_START, Config.BACKTEST_END)

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        log_returns = np.log1p(benchmark_returns)
        benchmark_return = np.expm1(log_returns.sum())
        benchmark_return_pct = benchmark_return * 100

        # 计算基准的最大回撤
        benchmark_cum = (1 + benchmark_returns).cumprod()
        benchmark_peak = benchmark_cum.expanding().max()
        benchmark_drawdown = (benchmark_cum - benchmark_peak) / benchmark_peak
        benchmark_max_dd = benchmark_drawdown.min() * 100

        # 计算基准的夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        excess_daily = benchmark_returns - risk_free_rate / 252
        benchmark_sharpe = (excess_daily.mean() / excess_daily.std()) * np.sqrt(252) if excess_daily.std() != 0 else 0

        # 计算基准的索提诺比率
        downside = excess_daily[excess_daily < 0]
        benchmark_sortino = (excess_daily.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() != 0 else 0

        benchmark_data = {
            'return': benchmark_return_pct,
            'max_dd': benchmark_max_dd,
            'sharpe': benchmark_sharpe,
            'sortino': benchmark_sortino
        }

        # 保存基准日收益率供图表使用
        display_results.benchmark_returns = benchmark_returns
    else:
        print("  ⚠ 无法计算等权基准收益率")
        benchmark_data = {'return': 0.0, 'max_dd': 0.0, 'sharpe': 0.0, 'sortino': 0.0}

    # 计算超额收益率
    excess_return_pct = m.total_return_pct - benchmark_data.get('return', 0)

    # ---- 策略表现框 ----
    print(f"\n  【策略表现】")
    print(f"  {'-' * 40}")
    print(f"  {'交易总次数':<14}: {int(m.trade_count):>10}")
    print(f"  {'总收益率':<14}: {m.total_return_pct:>10.2f}%")
    print(f"  {'超额收益率':<14}: {excess_return_pct:>10.2f}%")
    print(f"  {'最大回撤':<14}: {m.max_drawdown_pct:>10.2f}%")
    print(f"  {'夏普比率':<14}: {m.sharpe:>10.4f}")
    print(f"  {'索提诺比率':<14}: {m.sortino:>10.4f}")
    print(f"  {'盈亏比':<14}: {m.profit_factor:>10.4f}")
    print(f"  {'胜率':<14}: {m.win_rate:>10.2f}%")
    print(f"  {'总盈亏':<14}: {m.total_pnl:>12,.2f} 元")
    print(f"  {'总手续费':<14}: {m.total_fees:>12,.2f} 元")
    if m.initial_market_value and m.end_market_value:
        print(f"  {'初始市值':<14}: {m.initial_market_value:>12,.2f} 元")
        print(f"  {'最终市值':<14}: {m.end_market_value:>12,.2f} 元")

    # ---- 基准表现框 ----
    print(f"\n  【基准表现 - 成分股等权平均】")
    print(f"  {'-' * 40}")
    print(f"  {'基准收益率':<14}: {benchmark_data.get('return', 0):>10.2f}%")
    print(f"  {'最大回撤':<14}: {benchmark_data.get('max_dd', 0):>10.2f}%")
    print(f"  {'夏普比率':<14}: {benchmark_data.get('sharpe', 0):>10.4f}")
    print(f"  {'索提诺比率':<14}: {benchmark_data.get('sortino', 0):>10.4f}")

    # 显示交易记录汇总
    orders = result.orders
    if orders is not None and not orders.empty:
        buy_orders = orders[orders['type'] == 'buy']
        sell_orders = orders[orders['type'] == 'sell']
        logger.info("交易记录统计:")
        logger.info(f"买入次数: {len(buy_orders)}")
        logger.info(f"卖出次数: {len(sell_orders)}")
    else:
        logger.warning("未产生任何交易")


# ============================================================
# 模块8：结果持久化
# ============================================================

def save_results(result, df, output_dir=None):
    """
    将回测结果持久化保存到文件。
    
    保存内容：
      1. 回测指标摘要 (JSON)
      2. 交易记录 (CSV)
      3. 每日持仓市值 (CSV)
      4. 每日选股列表 (CSV)
    
    参数:
        result: pybroker.TestResult 对象
        df: 包含回测数据的 DataFrame
        output_dir: 输出目录，默认在项目根目录的results文件夹
    """
    if result is None:
        logger.error("无回测结果可保存")
        return
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # 1. 保存回测指标摘要
        m = result.metrics
        summary = {
            'backtest_time': timestamp,
            'backtest_start': Config.BACKTEST_START,
            'backtest_end': Config.BACKTEST_END,
            'initial_cash': Config.INITIAL_CASH,
            'top_n_stocks': Config.TOP_N_STOCKS,
            'trade_count': int(m.trade_count),
            'total_return_pct': float(m.total_return_pct),
            'max_drawdown_pct': float(m.max_drawdown_pct),
            'sharpe': float(m.sharpe),
            'sortino': float(m.sortino),
            'profit_factor': float(m.profit_factor),
            'win_rate': float(m.win_rate),
            'total_pnl': float(m.total_pnl),
            'total_fees': float(m.total_fees),
        }
        
        summary_path = os.path.join(output_dir, f'summary_{timestamp}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"回测摘要已保存: {summary_path}")
        
        # 2. 保存交易记录
        if result.orders is not None and not result.orders.empty:
            orders_path = os.path.join(output_dir, f'orders_{timestamp}.csv')
            result.orders.to_csv(orders_path, index=False, encoding='utf-8-sig')
            logger.info(f"交易记录已保存: {orders_path}")
        
        # 3. 保存每日持仓市值
        if result.portfolio is not None and not result.portfolio.empty:
            portfolio_path = os.path.join(output_dir, f'portfolio_{timestamp}.csv')
            portfolio_df = result.portfolio.reset_index()
            portfolio_df.to_csv(portfolio_path, index=False, encoding='utf-8-sig')
            logger.info(f"持仓市值已保存: {portfolio_path}")
        
        # 4. 保存每日选股列表
        if hasattr(factors, 'SELECTION_MAP') and factors.SELECTION_MAP:
            selections = []
            for date, symbols in factors.SELECTION_MAP.items():
                for sym in symbols:
                    selections.append({'date': date, 'symbol': sym})
            if selections:
                selections_df = pd.DataFrame(selections)
                selections_path = os.path.join(output_dir, f'selections_{timestamp}.csv')
                selections_df.to_csv(selections_path, index=False, encoding='utf-8-sig')
                logger.info(f"选股列表已保存: {selections_path}")
        
        logger.info(f"所有结果已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


# ============================================================
# 模块9：可视化分析
# ============================================================

def plot_results(result, df):
    """
    绘制回测结果的可视化图表。

    包含子图：
      (1) 收益曲线：策略净值 vs 基准
      (2) 回撤曲线
      (3) 月度收益热力图
      (4) 年度收益柱状图

    Args:
        result: pybroker.TestResult 对象
        df: 包含回测数据的 DataFrame（用于计算基准收益）
    """
    if result is None:
        print("  ✗ 无回测结果，跳过绘图")
        return

    print(f"\n[图表] 正在生成可视化图表...")

    # 获取账户净值曲线
    try:
        portfolio_df = result.portfolio
        if portfolio_df is None or portfolio_df.empty:
            print("  ✗ 无法获取投资组合数据，跳过绘图")
            return
    except Exception:
        print("  ✗ 无法获取投资组合数据，跳过绘图")
        return

    # ---- 计算基准收益（成分股等权平均） ----
    # 优先使用 display_results 已计算的等权基准，避免重复计算
    if hasattr(display_results, 'benchmark_returns') and display_results.benchmark_returns is not None:
        benchmark_returns = display_results.benchmark_returns
    else:
        benchmark_returns = _compute_equal_weight_benchmark(df, Config.BACKTEST_START, Config.BACKTEST_END)
    benchmark_returns = benchmark_returns.sort_index()

    # ---- 准备绘图数据 ----
    # result.portfolio 的 date 是 index 而非列名，需要 reset_index
    portfolio_df = portfolio_df.copy()
    portfolio_df = portfolio_df.reset_index()  # date 从 index 变为列
    if 'date' not in portfolio_df.columns and 'index' in portfolio_df.columns:
        portfolio_df.rename(columns={'index': 'date'}, inplace=True)
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])

    # 计算净值曲线
    portfolio_df['strategy_nav'] = portfolio_df['market_value'] / Config.INITIAL_CASH

    # 计算日收益率
    portfolio_df['daily_return'] = portfolio_df['market_value'].pct_change()

    # 计算回撤
    portfolio_df['cummax'] = portfolio_df['market_value'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['market_value'] - portfolio_df['cummax']) / portfolio_df['cummax']

    # 对齐基准数据
    benchmark_aligned = benchmark_returns.reindex(
        portfolio_df['date'], method=None
    ).fillna(0)

    # ---- 创建图表 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('量化多因子选股策略 - 回测结果分析', fontsize=16, fontweight='bold')

    # ===== 子图1: 收益曲线 =====
    ax1 = axes[0, 0]
    ax1.plot(portfolio_df['date'], portfolio_df['strategy_nav'],
             label='策略净值', color='#2196F3', linewidth=1.5)

    # 基准净值（从累计收益反推）
    benchmark_cum = (1 + benchmark_aligned).cumprod()
    ax1.plot(portfolio_df['date'], benchmark_cum.values,
             label='基准 (成分股等权)', color='#FF9800', linewidth=1, linestyle='--', alpha=0.7)

    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('净值曲线', fontsize=13, fontweight='bold')
    ax1.set_ylabel('净值')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ===== 子图2: 回撤曲线 =====
    ax2 = axes[0, 1]
    ax2.fill_between(portfolio_df['date'], 0, portfolio_df['drawdown'] * 100,
                     color='#E53935', alpha=0.3, label='回撤')
    ax2.plot(portfolio_df['date'], portfolio_df['drawdown'] * 100,
             color='#E53935', linewidth=0.8)
    ax2.set_title('回撤曲线', fontsize=13, fontweight='bold')
    ax2.set_ylabel('回撤 (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    # 反转Y轴使回撤向下
    ax2.invert_yaxis()

    # ===== 子图3: 月度收益热力图 =====
    ax3 = axes[1, 0]
    # 计算月度收益
    monthly_returns = portfolio_df.set_index('date')['daily_return'].resample('ME').apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    monthly_returns = monthly_returns.dropna()

    if len(monthly_returns) > 1:
        # 创建月度收益矩阵
        monthly_matrix = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        pivot = monthly_matrix.pivot_table(
            values='return', index='year', columns='month', aggfunc='mean'
        )

        im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels([f'{int(m)}月' for m in pivot.columns])
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels([str(int(y)) for y in pivot.index])
        ax3.set_title('月度收益热力图 (%)', fontsize=13, fontweight='bold')

        # 在热力图上标注数值
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > 5 else 'black'
                    ax3.text(j, i, f'{val:.1f}', ha='center', va='center',
                            color=text_color, fontsize=8)

        fig.colorbar(im, ax=ax3, shrink=0.8)
    else:
        ax3.text(0.5, 0.5, '月度数据不足', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('月度收益热力图', fontsize=13, fontweight='bold')

    # ===== 子图4: 年度收益/关键统计 =====
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_title('策略关键统计', fontsize=13, fontweight='bold')

    # 提取关键统计
    m = result.metrics
    stats_text = ""

    try:
        total_return = m.total_return_pct
        max_dd = m.max_drawdown_pct
        sharpe = m.sharpe
        win_rate = m.win_rate
        trade_count = int(m.trade_count)
        pf = m.profit_factor

        start_val = portfolio_df['market_value'].iloc[0]
        end_val = portfolio_df['market_value'].iloc[-1]

        stats_text = f"""
        ┌──────────────────────────┐
        │   策略核心指标           │
        ├──────────────────────────┤
        │  初始资金:   {Config.INITIAL_CASH:>12,} 元  │
        │  最终市值:   {end_val:>12,.0f} 元  │
        │  总收益率:   {total_return:>10.2f}%     │
        │  最大回撤:   {max_dd:>10.2f}%     │
        │  夏普比率:   {sharpe:>10.4f}        │
        │  交易次数:   {trade_count:>10}        │
        │  胜　率:     {win_rate:>10.2f}%     │
        │  盈亏比:     {pf:>10.4f}        │
        ├──────────────────────────┤
        │  持仓上限:   {Config.TOP_N_STOCKS:>10} 只    │
        │  选股范围:   中证500         │
        └──────────────────────────┘
        """
    except Exception:
        stats_text = "统计信息暂不可用"

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='SimHei',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ---- 保存并显示图表 ----
    plt.tight_layout()
    charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '图表')
    os.makedirs(charts_dir, exist_ok=True)
    chart_filename = datetime.now().strftime('%Y%m%d-%H%M%S') + '.png'
    output_path = os.path.join(charts_dir, chart_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存到: {output_path}")
    plt.show(block=False)
    plt.pause(0.5)
