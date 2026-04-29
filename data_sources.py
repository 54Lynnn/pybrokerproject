# -*- coding: utf-8 -*-
"""数据获取: akshare成分股 + baostock行情 + SQLite缓存"""

from config import Config
from datetime import datetime, timedelta
import akshare as ak
import baostock as bs
import os
import pandas as pd
import pybroker as pyb
import socket
import sqlite3
import threading
import time

def get_zz500_stocks():
    """
    获取中证500成分股列表（带线程超时保护）。

    中证500指数（000905）成分股，源自 akshare 的 index_stock_cons_csindex。
    该接口偶发网络卡死（不抛异常直接hang），因此使用子线程 + 超时方案。
    """
    print("\n[数据] 正在通过 akshare 获取中证500成分股列表...")

    max_retries = 3
    for attempt in range(max_retries):
        result = {}
        t = threading.Thread(target=lambda: result.update(
            {'df': ak.index_stock_cons_csindex(symbol=Config.INDEX_CODE)}
        ), daemon=True)
        t.start()
        t.join(timeout=60)

        if t.is_alive():
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"  ⚠ akshare 超时（{attempt+1}/{max_retries}），{wait}秒后重试...")
                time.sleep(wait)
                continue
            print(f"  ✗ akshare 多次超时，请稍后重试")
            return None

        if 'df' not in result:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"  ⚠ 获取失败（{attempt+1}/{max_retries}），{wait}秒后重试...")
                time.sleep(wait)
                continue
            print(f"  ✗ 获取中证500成分股失败")
            return None

        df = result['df']
        stocks = df[['成分券代码', '成分券名称']].copy()
        stocks.columns = ['code', 'name']
        stocks = stocks.drop_duplicates(subset='code', keep='first')
        stocks = stocks.reset_index(drop=True)

        print(f"  ✓ 成功获取 {len(stocks)} 只中证500成分股")

        if Config.STOCK_LIMIT is not None and len(stocks) > Config.STOCK_LIMIT:
            stocks = stocks.head(Config.STOCK_LIMIT)
            print(f"  ⚠ 已限制为前 {Config.STOCK_LIMIT} 只股票（测试模式）")

        return stocks

    return None


def get_stock_industry(stocks_df):
    """
    获取股票所属行业信息（使用akshare的stock_board_industry_ths接口）。

    返回：{code: industry_name} 字典
    """
    print("\n[数据] 正在获取股票行业信息...")

    industry_map = {}
    max_retries = 3

    for attempt in range(max_retries):
        result = {}
        t = threading.Thread(target=lambda: result.update(
            {'df': ak.stock_board_industry_name_ths()}
        ), daemon=True)
        t.start()
        t.join(timeout=60)

        if t.is_alive() or 'df' not in result:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"  ⚠ 行业数据获取超时（{attempt+1}/{max_retries}），{wait}秒后重试...")
                time.sleep(wait)
                continue
            print(f"  ⚠ 无法获取行业数据，将跳过行业中性化")
            return {}

        industry_df = result['df']
        break

    try:
        # stock_board_industry_name_ths 返回的列可能是 'name', 'code' 或 '板块名称', '板块代码'
        # 我们需要获取每个行业的成分股
        name_col = 'name' if 'name' in industry_df.columns else '板块名称'
        if name_col in industry_df.columns:
            industry_names = industry_df[name_col].tolist()
            print(f"  发现 {len(industry_names)} 个行业板块，正在获取成分股...")

            for ind_name in industry_names:
                try:
                    cons_df = ak.stock_board_industry_cons_ths(symbol=ind_name)
                    if cons_df is not None and not cons_df.empty:
                        # 列名可能是 '代码' 或 '个股代码' 或 'name'
                        code_col = None
                        for col in ['代码', '个股代码', 'code']:
                            if col in cons_df.columns:
                                code_col = col
                                break
                        if code_col:
                            for _, row in cons_df.iterrows():
                                code = str(row[code_col]).zfill(6)
                                industry_map[code] = ind_name
                except Exception:
                    continue
        else:
            print(f"  ⚠ 行业数据格式不符合预期，列名: {list(industry_df.columns)}")
            return {}
    except Exception as e:
        print(f"  ⚠ 行业数据解析失败: {e}")
        return {}

    # 只保留我们需要的股票
    valid_codes = set(stocks_df['code'].tolist())
    industry_map = {k: v for k, v in industry_map.items() if k in valid_codes}

    print(f"  ✓ 行业数据获取完成：{len(industry_map)} 只股票")
    return industry_map


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
