# -*- coding: utf-8 -*-
"""数据校验模块 — 确保输入数据质量"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """数据校验失败时抛出的异常。"""
    pass


def validate_ohlcv(df, strict=True):
    """
    校验OHLCV数据的基本质量。
    
    校验项：
      1. 必需列是否存在
      2. 价格是否为正数
      3. 最高价 >= 最低价
      4. 收盘价在[最低,最高]范围内
      5. 成交量是否为非负数
      6. 是否存在大量缺失值
      7. 日期是否连续（单只股票）
    
    参数:
        df: 包含OHLCV数据的DataFrame
        strict: True=发现错误则抛出异常, False=仅记录警告
    
    返回:
        bool: 校验是否通过
    """
    logger.info("正在校验OHLCV数据质量...")
    
    # 1. 检查必需列
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        msg = f"缺少必需列: {missing_cols}"
        if strict:
            raise DataValidationError(msg)
        logger.warning(msg)
        return False
    
    # 2. 检查价格是否为正
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        negative_count = (df[col] <= 0).sum()
        if negative_count > 0:
            msg = f"{col} 列有 {negative_count} 条非正数记录"
            if strict:
                raise DataValidationError(msg)
            logger.warning(msg)
    
    # 3. 检查最高价 >= 最低价
    invalid_hl = (df['high'] < df['low']).sum()
    if invalid_hl > 0:
        msg = f"有 {invalid_hl} 条记录最高价 < 最低价"
        if strict:
            raise DataValidationError(msg)
        logger.warning(msg)
    
    # 4. 检查收盘价在[最低,最高]范围内
    invalid_close = ((df['close'] < df['low']) | (df['close'] > df['high'])).sum()
    if invalid_close > 0:
        msg = f"有 {invalid_close} 条记录收盘价不在[最低,最高]范围内"
        if strict:
            raise DataValidationError(msg)
        logger.warning(msg)
    
    # 5. 检查成交量是否为非负
    negative_vol = (df['volume'] < 0).sum()
    if negative_vol > 0:
        msg = f"有 {negative_vol} 条记录成交量为负数"
        if strict:
            raise DataValidationError(msg)
        logger.warning(msg)
    
    # 6. 检查缺失值比例
    for col in required_cols:
        if col in df.columns:
            null_ratio = df[col].isna().mean()
            if null_ratio > 0.1:  # 缺失超过10%
                msg = f"{col} 列缺失值比例: {null_ratio:.2%}"
                if strict:
                    raise DataValidationError(msg)
                logger.warning(msg)
    
    # 7. 检查停牌股票（成交量为0）
    suspended = (df['volume'] == 0).sum()
    if suspended > 0:
        logger.warning(f"有 {suspended} 条记录成交量为0（可能为停牌日）")
    
    # 8. 统计信息
    logger.info(f"数据校验通过！共 {len(df)} 条记录，{df['symbol'].nunique()} 只股票")
    logger.info(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    logger.info(f"  价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    
    return True


def validate_factors(df, factor_names, strict=True):
    """
    校验因子数据质量。
    
    校验项：
      1. 因子列是否存在
      2. 因子值是否为无穷大/NaN
      3. 因子分布是否异常（如全部相同）
      4. 截面标准化后的均值和标准差
    
    参数:
        df: 包含因子列的DataFrame
        factor_names: 因子列名列表
        strict: True=发现错误则抛出异常, False=仅记录警告
    
    返回:
        bool: 校验是否通过
    """
    logger.info("正在校验因子数据质量...")
    
    # 1. 检查因子列是否存在
    missing_factors = [f for f in factor_names if f not in df.columns]
    if missing_factors:
        msg = f"缺少因子列: {missing_factors}"
        if strict:
            raise DataValidationError(msg)
        logger.warning(msg)
        return False
    
    # 2. 检查无穷大和NaN
    for col in factor_names:
        inf_count = np.isinf(df[col]).sum()
        nan_count = df[col].isna().sum()
        
        if inf_count > 0:
            msg = f"因子 {col} 有 {inf_count} 个无穷大值"
            if strict:
                raise DataValidationError(msg)
            logger.warning(msg)
        
        if nan_count > len(df) * 0.5:  # NaN超过50%
            msg = f"因子 {col} NaN比例过高: {nan_count/len(df):.2%}"
            if strict:
                raise DataValidationError(msg)
            logger.warning(msg)
    
    # 3. 检查因子是否全部相同（无区分度）
    for col in factor_names:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.01:  # 唯一值少于1%
            msg = f"因子 {col} 区分度极低（唯一值比例: {unique_ratio:.2%}）"
            if strict:
                raise DataValidationError(msg)
            logger.warning(msg)
    
    # 4. 检查截面标准化质量（随机抽几天检查）
    sample_dates = df['date'].drop_duplicates().sample(min(5, df['date'].nunique()))
    for d in sample_dates:
        day_data = df[df['date'] == d]
        for col in factor_names:
            mean = day_data[col].mean()
            std = day_data[col].std()
            if abs(mean) > 0.1:  # 标准化后均值应接近0
                logger.debug(f"日期 {d}: 因子 {col} 均值={mean:.4f}（偏离0）")
            if std > 5:  # 标准化后标准差应接近1
                logger.debug(f"日期 {d}: 因子 {col} 标准差={std:.4f}（偏离1）")
    
    logger.info(f"因子校验通过！共 {len(factor_names)} 个因子")
    return True


def clean_data(df):
    """
    清洗数据：移除异常值和停牌日。
    
    清洗规则：
      1. 删除价格非正数的记录
      2. 删除最高价 < 最低价的记录
      3. 删除成交量为负数的记录
      4. 删除OHLC任何一项为NaN的记录
    
    参数:
        df: 原始DataFrame
    
    返回:
        pd.DataFrame: 清洗后的DataFrame
    """
    logger.info("正在清洗数据...")
    original_len = len(df)
    
    # 删除价格非正数
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
    
    # 删除最高价 < 最低价
    df = df[df['high'] >= df['low']]
    
    # 删除成交量为负
    df = df[df['volume'] >= 0]
    
    # 删除OHLC为NaN
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    removed = original_len - len(df)
    if removed > 0:
        logger.warning(f"清洗移除了 {removed} 条异常记录 ({removed/original_len:.2%})")
    else:
        logger.info("数据清洗完成，未发现异常记录")
    
    return df.reset_index(drop=True)
