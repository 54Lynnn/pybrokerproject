# -*- coding: utf-8 -*-
"""
============================================================
  基于 pybroker 的 A股 多因子选股策略（多头策略）
  数据来源：baostock（行情）+ akshare（指数成分股）
  回测框架：pybroker
  可视化：matplotlib
============================================================

策略流程：
  1. 通过 akshare 获取中证500指数成分股列表
  2. 通过 baostock 获取每只股票的日K线数据（最近2年）
  3. 数据缓存到本地 SQLite 数据库，避免重复下载
  4. 计算多个技术面和估值因子
  5. 每日对股票进行综合打分，选取排名靠前的股票
  6. 使用 pybroker 进行回测
  7. 可视化回测结果（收益曲线、回撤、月度收益等）

使用方式：
  在你的 Anaconda 环境中直接运行此文件即可：
  python E:/project/pybroker/量化多因子选股策略.py
"""


# ============================================================
# 模块1：环境配置与库导入
# ============================================================
# 所有必需的第三方库导入，确保 Anaconda 环境已安装这些库

import pybroker as pyb
from pybroker import ExecContext, StrategyConfig, Strategy
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

# 正常显示中文和负号（Windows下常见问题）
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# 设置 pybroker 数据源缓存（使用 diskcache，类似 SQLite）
pyb.enable_data_source_cache('akshare')
pyb.enable_data_source_cache('custom')

print("=" * 60)
print("  量化多因子选股策略 启动")
print(f"  pybroker 版本: {pyb.__version__}")
print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


# ============================================================
# 模块2：全局配置参数
# ============================================================
# 将所有可调参数集中在此处，方便修改和优化

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
    MA_SHORT = 5                        # 短期均线周期
    MA_LONG = 20                        # 长期均线周期
    RSI_PERIOD = 14                     # RSI 计算周期
    VOLUME_MA_PERIOD = 20               # 成交量均线周期
    MOMENTUM_PERIOD = 20                # 动量计算周期

    # -------- 选股参数 --------
    TOP_N_STOCKS = 5                    # 每日选股数量（持仓股票数上限）
    FACTOR_WEIGHTS = {                  # 各因子权重（总和为1）
        'earnings_yield': 0.25,         # 盈利收益率（1/PE）
        'book_yield': 0.20,             # 净资产收益率（1/PB）
        'momentum_20d': 0.20,           # 20日动量
        'volume_ratio': 0.15,           # 量比
        'rsi_score': 0.20,              # RSI 信号得分
    }

    # -------- 资金与风控参数 --------
    INITIAL_CASH = 1_000_000            # 初始资金（元）
    STOP_LOSS_PCT = -0.10               # 止损线：亏损10%止损
    TAKE_PROFIT_PCT = 0.30              # 止盈线：盈利30%止盈
    A_SHARE_LOT = 100                   # A股每手100股

    # -------- 回测参数 --------
    # BACKTEST_START 默认值在 main() 中根据实际 DATA_START_DATE 动态计算
    BACKTEST_START = _yesterday           # 占位，main() 中会覆写
    BACKTEST_END = _yesterday             # 回测结束（昨天）


# ============================================================
# 模块3：获取中证500成分股列表（akshare）
# ============================================================

def get_zz500_stocks():
    """
    通过 akshare 获取中证500指数的成分股列表。

    中证500指数（000905）由全部A股中剔除沪深300指数成分股及总市值排名前300名
    的股票后，总市值排名靠前的500只股票组成，综合反映A股市场中等市值公司的表现。

    Returns:
        pd.DataFrame: 包含 'code'（股票代码）和 'name'（股票名称）两列的 DataFrame
    """
    print("\n[数据] 正在通过 akshare 获取中证500成分股列表...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = ak.index_stock_cons_csindex(symbol=Config.INDEX_CODE)

            stocks = df[['成分券代码', '成分券名称']].copy()
            stocks.columns = ['code', 'name']

            stocks = stocks.drop_duplicates(subset='code', keep='first')
            stocks = stocks.reset_index(drop=True)

            print(f"  ✓ 成功获取 {len(stocks)} 只中证500成分股")

            if Config.STOCK_LIMIT is not None and len(stocks) > Config.STOCK_LIMIT:
                stocks = stocks.head(Config.STOCK_LIMIT)
                print(f"  ⚠ 已限制为前 {Config.STOCK_LIMIT} 只股票（测试模式）")

            return stocks

        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"  ⚠ 获取失败（{attempt+1}/{max_retries}），{wait}秒后重试: {e}")
                time.sleep(wait)
            else:
                print(f"  ✗ 获取中证500成分股失败: {e}")
                if os.path.exists(Config.SQLITE_DB_PATH):
                    print(f"  提示：数据库缓存仍可用，可修改 --start/--end 后重试")
                return None

    return None


# ============================================================
# 模块4：数据获取与 SQLite 缓存（baostock）
# ============================================================

def convert_code_to_baostock(code):
    """
    将标准股票代码转换为 baostock 格式。

    baostock 的股票代码格式为 "sh.600000" 或 "sz.000001"，
    需要根据股票代码的第一个数字来判断交易所：
      - 6xxxxx → 上海交易所 → sh.6xxxxx
      - 0xxxxx, 3xxxxx → 深圳交易所 → sz.0xxxxx / sz.3xxxxx
      - 4xxxxx, 8xxxxx → 北京交易所 → bj.4xxxxx / bj.8xxxxx

    Args:
        code: 6位股票代码字符串，如 "600000"

    Returns:
        str: baostock 格式的股票代码，如 "sh.600000"
    """
    code = str(code).zfill(6)       # 确保是6位字符串
    first_char = code[0]

    if first_char == '6':
        return f'sh.{code}'
    elif first_char in ('0', '3'):
        return f'sz.{code}'
    elif first_char in ('4', '8'):
        return f'bj.{code}'
    else:
        return f'sz.{code}'         # 默认深圳


def init_sqlite_db():
    """
    初始化 SQLite 数据库，创建日K线数据表。

    如果数据表已存在，则不重复创建。
    使用 SQLite 而非 CSV 的优势：
      1. 体积更小（数据压缩存储）
      2. 查询更快（支持索引和按条件筛选）
      3. 并发安全（文件锁机制）
      4. 数据类型保真（不会出现CSV的编码/精度问题）
    """
    db_path = Config.SQLITE_DB_PATH

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_kline (
                code         TEXT    NOT NULL,   -- 股票代码（baostock格式）
                date         TEXT    NOT NULL,   -- 交易日期 YYYY-MM-DD
                open         REAL,               -- 开盘价
                high         REAL,               -- 最高价
                low          REAL,               -- 最低价
                close        REAL,               -- 收盘价
                preclose     REAL,               -- 前收盘价
                volume       REAL,               -- 成交量（股）
                amount       REAL,               -- 成交额（元）
                turn         REAL,               -- 换手率（%）
                pctChg       REAL,               -- 涨跌幅（%）
                peTTM        REAL,               -- 滚动市盈率
                pbMRQ        REAL,               -- 市净率
                psTTM        REAL,               -- 滚动市销率
                pcfNcfTTM    REAL,               -- 滚动市现率
                PRIMARY KEY (code, date)
            )
        """)
        # 创建索引以加速查询
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_code_date
            ON daily_kline(code, date)
        """)
        conn.commit()

    print(f"\n[数据库] SQLite 数据库已就绪: {db_path}")


def check_data_in_db(stock_code_bs):
    """
    检查数据库中某只股票的数据是否足够用于回测。

    判断标准：
      1. 请求区间内的数据条数 >= 100
      2. 数据最早日期 <= 请求起始日期 + 14天容差
      3. 数据最晚日期 >= 请求结束日期 - 14天容差

    14天容差 ≈ 10个交易日，边界上差几天不影响指标预热。
    """
    db_path = Config.SQLITE_DB_PATH
    if not os.path.exists(db_path):
        return False

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM daily_kline WHERE code = ?",
            (stock_code_bs,)
        )
        count, min_date, max_date = cursor.fetchone()

        if count is None or count < 100 or min_date is None:
            return False

        # 14天日期容差（≈10个交易日）
        tolerance = timedelta(days=14)
        data_start_ok = pd.Timestamp(min_date) <= pd.Timestamp(Config.DATA_START_DATE) + tolerance
        data_end_ok = max_date is None or pd.Timestamp(max_date) >= pd.Timestamp(Config.DATA_END_DATE) - tolerance

        return data_start_ok and data_end_ok


def _download_bs_raw(bs_code, start_date, end_date, result_container):
    """
    在子线程中执行 baostock 查询。
    """
    socket.setdefaulttimeout(10)
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM',
            start_date=start_date,
            end_date=end_date,
            frequency='d',
            adjustflag='2'
        )
        if rs.error_code != '0':
            result_container['error'] = rs.error_msg
            return

        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())

        if data_list:
            result_container['df'] = pd.DataFrame(data_list, columns=rs.fields)
    except Exception as e:
        result_container['error'] = str(e)


def download_single_stock(bs_code, max_retries=2,
                         start_date=None, end_date=None):
    """
    从 baostock 下载单只股票的日K线数据，带线程超时和重试机制。

    Args:
        bs_code: baostock 格式的股票代码
        max_retries: 最大重试次数
        start_date: 下载起始日期（默认取 Config.DATA_START_DATE）
        end_date: 下载结束日期（默认取 Config.DATA_END_DATE）

    Returns:
        pd.DataFrame: 下载成功返回数据，失败返回空DataFrame
    """
    if start_date is None:
        start_date = Config.DATA_START_DATE
    if end_date is None:
        end_date = Config.DATA_END_DATE

    DOWNLOAD_TIMEOUT = 12

    for attempt in range(max_retries):
        result = {}
        thread = threading.Thread(
            target=_download_bs_raw,
            args=(bs_code, start_date, end_date, result),
            daemon=True
        )
        thread.start()
        thread.join(timeout=DOWNLOAD_TIMEOUT)

        if thread.is_alive():
            # 子线程超时未返回，说明 baostock socket 卡住了
            if attempt < max_retries - 1:
                print(f"    ⚠ [{bs_code}] 超时({DOWNLOAD_TIMEOUT}s)，第{attempt+1}次重试...")
                time.sleep(3)
                continue
            print(f"    ✗ [{bs_code}] 多次超时，跳过")
            return pd.DataFrame()

        if 'df' in result and result['df'] is not None:
            return result['df']

        if 'error' in result and result['error']:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return pd.DataFrame()

        # 空数据
        return pd.DataFrame()

    return pd.DataFrame()


def save_to_sqlite(df, bs_code):
    """
    将单只股票的日K线数据保存到 SQLite 数据库。

    只插入数据库中不存在的日期，已有日期的不覆盖。
    这样即使下载了全区间数据，也不会重复插入已有的行。
    """
    if df.empty:
        return

    db_path = Config.SQLITE_DB_PATH

    numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume',
                    'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    cols_to_save = ['date', 'code'] + numeric_cols
    cols_available = [c for c in cols_to_save if c in df.columns]
    df_to_save = df[cols_available].copy()

    with sqlite3.connect(db_path) as conn:
        # 查询数据库中已有日期，只插入新日期
        existing = conn.execute(
            "SELECT date FROM daily_kline WHERE code = ?", (bs_code,)
        ).fetchall()
        existing_dates = {row[0] for row in existing} if existing else set()

        new_rows = df_to_save[~df_to_save['date'].isin(existing_dates)]
        if not new_rows.empty:
            new_rows.to_sql('daily_kline', conn, if_exists='append', index=False)


def download_all_stocks_data(stocks_df):
    """
    批量下载所有成分股的日K线数据。

    下载流程：
      1. 登录 baostock
      2. 遍历每只股票，检查数据库中是否已有数据
      3. 如果没有或不够，则从 baostock 下载
      4. 保存到 SQLite 数据库
      5. 添加延迟以避免被 baostock 限制
      6. 登出 baostock

    Args:
        stocks_df: 包含 'code' 列的成分股 DataFrame

    Note:
        首次运行时，500只股票 × 每只约0.5秒 ≈ 4分钟
        之后从 SQLite 读取，几乎瞬间完成
    """
    print(f"\n[数据] 开始下载 {len(stocks_df)} 只股票的日K线数据...")
    print(f"  时间范围: {Config.DATA_START_DATE} ~ {Config.DATA_END_DATE}")
    print(f"  (每次运行只下载新数据，已有数据从 SQLite 缓存读取)")

    # 登录 baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"  ✗ baostock 登录失败: {lg.error_msg}")
        print(f"  提示：已有缓存数据可直接用于回测")
        return
    print(f"  ✓ baostock 登录成功")

    # 初始化 SQLite 数据库
    init_sqlite_db()

    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_stocks = []           # 记录失败的股票（名称+原因）
    consecutive_fails = 0     # 连续失败计数器
    total = len(stocks_df)

    for i, row in stocks_df.iterrows():
        idx = i
        stock_code = row['code']
        stock_name = row['name']
        bs_code = convert_code_to_baostock(stock_code)

        # 进度显示：每5%打印一次
        pct = (idx + 1) / total * 100
        if idx % max(1, total // 20) == 0:
            print(f"  [{idx+1:>3}/{total}] {pct:5.1f}%  "
                  f"新: {success_count}  存: {skip_count}  缺: {fail_count}  "
                  f"当前: {stock_code} {stock_name}")

        # 检查数据库覆盖情况 → 只下载缺失的日期范围
        if check_data_in_db(bs_code):
            skip_count += 1
            continue

        # 下载缺失部分数据（内置线程超时保护，单只最长12秒）
        df = download_single_stock(bs_code)
        if not df.empty:
            save_to_sqlite(df, bs_code)
            success_count += 1
            consecutive_fails = 0     # 重置连续失败计数
        else:
            fail_count += 1
            consecutive_fails += 1
            failed_stocks.append(f"{stock_code} {stock_name}")
            if fail_count <= 10:
                print(f"  ⚠ [{idx+1}/{total}] {stock_code} {stock_name} 无数据（可能尚未上市或已退市）")

            # 连续失败超过15次，尝试重新登录 baostock
            if consecutive_fails >= 15:
                print(f"  ⟳ 连续失败{consecutive_fails}次，重新登录 baostock...")
                time.sleep(3)
                try:
                    bs.logout()
                except Exception:
                    pass
                time.sleep(1)
                lg = bs.login()
                if lg.error_code == '0':
                    print(f"  ✓ 重新登录成功，继续")
                    consecutive_fails = 0
                else:
                    print(f"  ✗ 重连失败，剩余股票将跳过下载")
                    break

        # 下载间隔
        time.sleep(0.3)

    # 登出 baostock
    try:
        bs.logout()
    except Exception:
        pass

    print(f"\n  下载完成！")
    print(f"    ✓ 新下载: {success_count} 只")
    print(f"    → 已缓存: {skip_count} 只")
    print(f"    ✗ 失败:   {fail_count} 只")
    print(f"  数据存储位置: {Config.SQLITE_DB_PATH}")


def load_kline_from_sqlite(stocks_df):
    """
    从 SQLite 数据库加载所有股票的日K线数据到 pandas DataFrame。

    返回的 DataFrame 格式与 pybroker 要求兼容：
      - 列：symbol, date, open, high, low, close, volume
      - date 列为 datetime 类型
      - 按 symbol + date 排序

    Args:
        stocks_df: 成分股列表（用于限制加载范围）

    Returns:
        pd.DataFrame: 包含所有股票日K线数据的 DataFrame
    """
    print(f"\n[数据] 从 SQLite 加载日K线数据...")

    db_path = Config.SQLITE_DB_PATH
    if not os.path.exists(db_path):
        print("  ✗ 数据库文件不存在！")
        return pd.DataFrame()

    # 构建股票代码列表
    codes = [convert_code_to_baostock(str(c)) for c in stocks_df['code'].tolist()]

    with sqlite3.connect(db_path) as conn:
        # 只加载我们需要的股票和时间范围
        query = """
            SELECT code, date, open, high, low, close, volume, amount, turn, pctChg,
                   peTTM, pbMRQ, psTTM, pcfNcfTTM
            FROM daily_kline
            WHERE code IN ({})
              AND date >= ?
              AND date <= ?
            ORDER BY code, date
        """.format(','.join(['?'] * len(codes)))

        params = codes + [Config.DATA_START_DATE, Config.DATA_END_DATE]
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        print("  ✗ 没有找到任何数据！")
        return df

    # 列名重命名：code → symbol（pybroker 要求）
    df.rename(columns={'code': 'symbol'}, inplace=True)

    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'])

    # 转换为正确的数值类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                    'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 剔除停牌日（成交量为0的交易日通常是停牌）
    df = df[df['volume'] > 0].copy()

    # 确保按 symbol + date 排序
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    print(f"  ✓ 加载完成：{df['symbol'].nunique()} 只股票，{len(df)} 条日K线记录")
    print(f"    日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

    return df


# ============================================================
# 模块5：技术指标与多因子计算
# ============================================================

def compute_indicators_for_group(group_df):
    """
    对单只股票的时间序列数据计算所有技术指标和因子。

    此函数会被 pandas groupby 对每只股票分别调用。
    计算内容包括：RSI、动量、量比、均线偏离等。

    Args:
        group_df: 单只股票的时间序列 DataFrame（已按日期升序排列）

    Returns:
        pd.DataFrame: 添加了指标列的 DataFrame
    """
    group_df = group_df.copy().sort_values('date').reset_index(drop=True)
    close = group_df['close'].values.astype(float)
    volume = group_df['volume'].values.astype(float)

    # ---- RSI（相对强弱指标）----
    # RSI = 100 - 100/(1 + RS)，其中 RS = N日内上涨平均值 / N日内下跌平均值
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # 使用指数移动平均平滑
    avg_gain = pd.Series(gain).ewm(span=Config.RSI_PERIOD, min_periods=Config.RSI_PERIOD, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=Config.RSI_PERIOD, min_periods=Config.RSI_PERIOD, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    group_df['rsi'] = 100 - 100 / (1 + rs)

    # ---- 动量（N日收益率）----
    # 动量反映股票的近期价格趋势
    group_df['return_5d'] = group_df['close'] / group_df['close'].shift(5) - 1     # 5日收益率
    group_df['return_20d'] = group_df['close'] / group_df['close'].shift(20) - 1   # 20日收益率

    # ---- 量比（成交量 / N日均量）----
    # 量比 > 1 表示当日成交活跃，可能伴随行情变化
    vol_ma = pd.Series(volume).rolling(window=Config.VOLUME_MA_PERIOD, min_periods=1).mean().values
    group_df['volume_ratio'] = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma > 0)

    # ---- 均线 ----
    group_df['ma_5'] = group_df['close'].rolling(window=Config.MA_SHORT, min_periods=1).mean()
    group_df['ma_20'] = group_df['close'].rolling(window=Config.MA_LONG, min_periods=1).mean()

    # ---- 波动率（20日年化波动率）----
    group_df['volatility_20d'] = group_df['pctChg'].rolling(window=20, min_periods=5).std()

    return group_df


def compute_all_indicators(df):
    """
    对所有股票的 DataFrame 分组计算技术指标。

    使用手动分组循环代替 groupby.apply，避免不同 pandas 版本
    中 include_groups=False 等参数导致的列丢失问题。

    Args:
        df: 原始日K线 DataFrame

    Returns:
        pd.DataFrame: 添加了所有技术指标列的 DataFrame
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
    print(f"  ✓ 指标计算完成：RSI、动量、量比、均线、波动率")
    return df


def compute_factor_scores(df):
    """
    计算每个交易日每只股票的综合因子得分。

    因子体系（共5个因子）：
      Factor 1 - 盈利收益率 (earnings_yield): 1/PE_TTM，越高表示估值越低越好
      Factor 2 - 净资产收益率 (book_yield): 1/PB_MRQ，越高表示估值越低越好
      Factor 3 - 20日动量 (momentum_20d): 过去20个交易日收益率，越高越好
      Factor 4 - 量比 (volume_ratio): 当日成交量/20日均量，越高越活跃
      Factor 5 - RSI得分 (rsi_score): RSI接近超卖区域得分高，捕捉反弹机会

    处理流程：
      1. 对每个因子进行截面标准化（z-score），去除量纲差异
      2. 根据因子方向调整符号（正向因子保持正号，反向因子取反）
      3. 按权重加权求和得到综合得分

    Args:
        df: 包含所有指标和估值数据的 DataFrame

    Returns:
        pd.DataFrame: 添加了 'composite_score' 和 'rank' 列的 DataFrame
    """
    print(f"\n[因子] 正在计算多因子综合得分...")

    df = df.copy()

    # ===== Step 1: 计算原始因子值 =====

    # 估值因子：处理 PE/PB 为0或负值的情况（用NaN替代）
    pe = df['peTTM'].replace(0, np.nan).values
    pb = df['pbMRQ'].replace(0, np.nan).values

    # Factor 1: 盈利收益率 = 1/PE（PE越低，盈利收益率越高，越好）
    df['f_earnings_yield'] = np.where((pe > 0) & (~np.isnan(pe)), 1.0 / pe, np.nan)

    # Factor 2: 净资产收益率 = 1/PB（PB越低，净资产收益率越高，越好）
    df['f_book_yield'] = np.where((pb > 0) & (~np.isnan(pb)), 1.0 / pb, np.nan)

    # Factor 3: 20日动量（直接使用前期计算的 return_20d）
    df['f_momentum_20d'] = df['return_20d']

    # Factor 4: 量比（直接使用前期计算的 volume_ratio，裁剪极端值）
    df['f_volume_ratio'] = df['volume_ratio'].clip(0.1, 5.0)

    # Factor 5: RSI信号得分
    # 逻辑：RSI在30-40之间（超卖区附近）得分最高，RSI过高（超买）得分低
    # 使用二次函数：score = 100 - (RSI - 35)^2 / 200，峰值在RSI=35
    rsi = df['rsi'].fillna(50).clip(0, 100).values
    df['f_rsi_score'] = 100 - ((rsi - 35) ** 2) / 200
    df['f_rsi_score'] = df['f_rsi_score'].clip(0, 100)

    # ===== Step 2: 截面标准化（每个交易日对所有股票做 z-score） =====
    factor_cols = ['f_earnings_yield', 'f_book_yield', 'f_momentum_20d',
                   'f_volume_ratio', 'f_rsi_score']

    # 手动循环代替 groupby.apply，避免 pandas 版本兼容性问题
    print(f"\n  正在进行截面标准化（{df['date'].nunique()} 个交易日）...")
    result_list = []
    dates = sorted(df['date'].unique())
    for i, d in enumerate(dates):
        mask = df['date'] == d
        group = df[mask].copy()
        for col in factor_cols:
            std = group[col].std()
            if std == 0 or np.isnan(std):
                group[col] = 0.0
            else:
                group[col] = (group[col] - group[col].mean()) / std
        result_list.append(group)
        if (i + 1) % 100 == 0 or (i + 1) == len(dates):
            print(f"  标准化进度: {i+1}/{len(dates)} ({100*(i+1)/len(dates):.0f}%)")

    df = pd.concat(result_list, ignore_index=True)

    # ===== Step 3: 综合得分加权求和 =====
    weights = Config.FACTOR_WEIGHTS
    df['composite_score'] = 0.0

    df['composite_score'] += df['f_earnings_yield'].fillna(0) * weights['earnings_yield']
    df['composite_score'] += df['f_book_yield'].fillna(0) * weights['book_yield']
    df['composite_score'] += df['f_momentum_20d'].fillna(0) * weights['momentum_20d']
    df['composite_score'] += df['f_volume_ratio'].fillna(0) * weights['volume_ratio']
    df['composite_score'] += df['f_rsi_score'].fillna(0) * weights['rsi_score']

    # ===== Step 4: 每日排名 =====
    # rank=1 表示当天综合得分最高的股票
    df['rank'] = df.groupby('date')['composite_score'].rank(ascending=False, method='first')

    # ===== Step 5: 生成选股信号 =====
    # 排名在 TOP_N_STOCKS 以内的股票标记为 selected=1，否则为0
    df['selected'] = np.where(df['rank'] <= Config.TOP_N_STOCKS, 1, 0)

    print(f"  ✓ 因子得分计算完成，每日选取前 {Config.TOP_N_STOCKS} 只股票")
    return df


# ============================================================
# 模块6：pybroker 交易策略定义
# ============================================================

# ============================================================
# 模块6：pybroker 交易策略定义
# ============================================================
# 采用官方轮动交易模式：
#   before_exec (rank)  → 每日跨股票排名，选出 top_symbols
#   add_execution (execute) → 每股票根据 top_symbols 决定买卖
# 参考：https://www.pybroker.com/zh-cn/latest/notebooks/10.%20Rotational%20Trading.html

# 模块级变量：选股映射表，由 run_backtest 设置
SELECTION_MAP = {}


def build_daily_selections(df):
    """从因子打分结果构建每日选股映射表。"""
    selected_df = df[df['selected'] == 1][['date', 'symbol']].copy()
    selected_df['date'] = pd.to_datetime(selected_df['date'])
    selection_map = {}
    for date, group in selected_df.groupby('date'):
        selection_map[date] = set(group['symbol'].tolist())
    print(f"  ✓ 构建每日选股表：{len(selection_map)} 个交易日")
    return selection_map


def rank(ctxs: dict[str, ExecContext]):
    """
    before_exec —— 每个 bar 开始前，在所有活跃股票中找出今日选股。

    官方轮动模式：在 before_exec 中完成跨股票的排名/筛选，
    结果存入全局参数 pyb.param，供 exec 函数读取。

    逻辑：根据当前 bar 日期，从预计算的 SELECTION_MAP 中
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
    for _ in range(10):
        if target_date in SELECTION_MAP:
            break
        target_date -= pd.Timedelta(days=1)

    if target_date in SELECTION_MAP:
        top_list = list(SELECTION_MAP[target_date])
    else:
        top_list = []

    # 存入全局参数供 exec 函数读取
    pyb.param('top_symbols', top_list)
    pyb.param('target_size', 1.0 / Config.TOP_N_STOCKS)


def execute_strategy(ctx: ExecContext):
    """
    per-symbol 执行函数 —— 根据 before_exec 排出的 top_symbols 下单。

    官方轮动模式中的 rotate 函数：
      - 如果持仓且不在 top_symbols 中 → 全部卖出
      - 如果无持仓且在 top_symbols 中 → 按目标仓位买入

    关键 API（来自官方文档）：
      - ctx.long_pos() → Position 或 None
      - ctx.close[-1]  → 最新收盘价
      - ctx.sell_all_shares() → 清仓
      - ctx.calc_target_shares(target) → 计算目标持仓股数
      - ctx.buy_shares / ctx.buy_limit_price → 下单
    """
    top_symbols = pyb.param('top_symbols')
    target_size = pyb.param('target_size')

    pos = ctx.long_pos()

    # 情况1：有持仓，但已不在选股列表中 → 清仓
    if pos:
        if ctx.symbol not in top_symbols:
            ctx.sell_all_shares()
        return  # 继续持有，不操作

    # 情况2：无持仓，且在选股列表中 → 买入
    if ctx.symbol in top_symbols:
        ctx.buy_shares = ctx.calc_target_shares(target_size)
        ctx.buy_limit_price = ctx.close[-1]


# ============================================================
# 模块7：回测执行
# ============================================================

def run_backtest(df):
    """
    使用 pybroker 执行多因子选股策略回测（官方轮动交易模式）。

    架构（参考官方 Rotational Trading）：
      1. build_daily_selections → 构建 {date: {symbols}} 选股表
      2. strategy.set_before_exec(rank) → 每日跨股票排名
      3. strategy.add_execution(execute_strategy, symbols) → per-symbol 下单
      4. strategy.backtest(warmup=20) → 执行回测
    """
    global SELECTION_MAP

    print(f"\n{'=' * 60}")
    print(f"  开始 pybroker 回测")
    print(f"  初始资金: {Config.INITIAL_CASH:,.0f} 元")
    print(f"  回测区间: {Config.BACKTEST_START} ~ {Config.BACKTEST_END}")
    print(f"  持仓上限: {Config.TOP_N_STOCKS} 只")
    print(f"{'=' * 60}")

    # ---- 构建每日选股映射表 ----
    print(f"\n  构建选股信号...")
    SELECTION_MAP = build_daily_selections(df)

    if not SELECTION_MAP:
        print("  ✗ 选股信号为空！")
        return None

    # 初始化全局参数默认值
    pyb.param('top_symbols', [])
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
        print("  ✗ 回测数据为空！")
        return None

    print(f"  回测数据: {df_bt['symbol'].nunique()} 只股票, {len(df_bt)} 条记录")

    symbols = df_bt['symbol'].unique().tolist()

    # ---- 创建策略配置 ----
    config = StrategyConfig(
        initial_cash=Config.INITIAL_CASH,
        max_long_positions=Config.TOP_N_STOCKS,  # 最大同时持仓数
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
    print(f"\n  正在执行回测，请稍候...")
    result = strategy.backtest(
        warmup=20,
        disable_parallel=False,
    )

    print(f"  ✓ 回测完成！")
    return result


def display_results(result):
    """
    打印回测结果的核心指标。

    输出指标包括：
      - 交易次数、胜率
      - 总收益率、最大回撤
      - 夏普比率、索提诺比率
      - 盈亏比（Profit Factor）

    Args:
        result: pybroker.TestResult 对象
    """
    if result is None:
        print("  ✗ 无回测结果可展示")
        return

    print(f"\n{'=' * 60}")
    print(f"  回测结果")
    print(f"{'=' * 60}")

    # 使用 result.metrics (EvalMetrics dataclass) 直接取字段
    m = result.metrics

    print(f"  {'交易总次数':<14}: {int(m.trade_count):>10}")
    print(f"  {'总收益率':<14}: {m.total_return_pct:>10.2f}%")
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

    # 显示交易记录汇总
    orders = result.orders
    if orders is not None and not orders.empty:
        buy_orders = orders[orders['type'] == 'buy']
        sell_orders = orders[orders['type'] == 'sell']
        print(f"\n  交易记录统计:")
        print(f"    买入次数: {len(buy_orders)}")
        print(f"    卖出次数: {len(sell_orders)}")
    else:
        print(f"\n  ⚠ 未产生任何交易")


# ============================================================
# 模块8：可视化分析
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

    # ---- 计算基准收益（中证500等权） ----
    # 使用所有股票的平均每日涨跌幅作为基准
    benchmark_returns = df.groupby('date')['pctChg'].mean() / 100.0
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
             label='基准 (中证500等权)', color='#FF9800', linewidth=1, linestyle='--', alpha=0.7)

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


# ============================================================
# 主程序入口
# ============================================================

def parse_args():
    """
    解析命令行参数。

    用户只需指定回测区间（--start / --end），
    程序自动推导数据区间 = 回测起始 - DATA_LEAD_DAYS ~ 回测结束。

    用法示例：
      python 量化多因子选股策略.py
      python 量化多因子选股策略.py --start 2023-01-01 --end 2025-12-31
      python 量化多因子选股策略.py --stocks 30 --top-n 10 --cash 500000
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
        """
    )
    p.add_argument('--start', type=str, default=default_backtest_start,
                   help=f'回测起始日期 (默认: {default_backtest_start})')
    p.add_argument('--end', type=str, default=default_backtest_end,
                   help=f'回测结束日期 (默认: {default_backtest_end})')
    p.add_argument('--stocks', type=int, default=0,
                   help='股票数量上限 (0=全部500只, 默认: 0)')
    p.add_argument('--top-n', type=int, default=Config.TOP_N_STOCKS,
                   help=f'每日持仓数 (默认: {Config.TOP_N_STOCKS})')
    p.add_argument('--cash', type=int, default=Config.INITIAL_CASH,
                   help=f'初始资金/元 (默认: {Config.INITIAL_CASH})')
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
            print(f"  ✗ {name}日期格式错误: {date_str}")
            return

    print(f"\n  回测区间: {Config.BACKTEST_START} ~ {Config.BACKTEST_END}")
    print(f"  数据区间: {Config.DATA_START_DATE} ~ {Config.DATA_END_DATE}")
    print(f"  (数据比回测早 {Config.DATA_LEAD_DAYS} 天，用于指标预热)")

    print(f"\n{'=' * 60}")
    print(f"  步骤1: 获取中证500成分股列表")
    print(f"{'=' * 60}")
    stocks_df = get_zz500_stocks()

    if stocks_df is None or stocks_df.empty:
        print("\n  ✗ 无法获取成分股列表，回测终止")
        print("  提示：akshare 可能暂时不可用，稍后重试或检查网络连接")
        return

    print(f"\n{'=' * 60}")
    print(f"  步骤2: 下载/加载日K线数据")
    print(f"{'=' * 60}")
    # 先尝试下载数据（如有缓存则自动跳过）
    download_all_stocks_data(stocks_df)
    # 从 SQLite 加载数据到 DataFrame
    df = load_kline_from_sqlite(stocks_df)

    if df.empty:
        print("  ✗ 没有数据可供回测，程序退出")
        return

    print(f"\n{'=' * 60}")
    print(f"  步骤3: 计算技术指标")
    print(f"{'=' * 60}")
    df = compute_all_indicators(df)

    print(f"\n{'=' * 60}")
    print(f"  步骤4: 多因子打分与选股")
    print(f"{'=' * 60}")
    df = compute_factor_scores(df)

    print(f"\n{'=' * 60}")
    print(f"  步骤5: 执行回测")
    print(f"{'=' * 60}")
    result = run_backtest(df)

    print(f"\n{'=' * 60}")
    print(f"  步骤6: 展示结果")
    print(f"{'=' * 60}")
    display_results(result)

    print(f"\n{'=' * 60}")
    print(f"  步骤7: 可视化图表")
    print(f"{'=' * 60}")
    plot_results(result, df)

    print(f"\n{'=' * 60}")
    print(f"  策略运行完毕！")
    print(f"  数据库文件: {Config.SQLITE_DB_PATH}")
    print(f"  图表目录: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '图表')}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()