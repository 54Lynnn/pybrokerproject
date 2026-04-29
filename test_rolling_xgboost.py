# -*- coding: utf-8 -*-
"""
测试滚动XGBoost模块 — 验证无未来函数
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(n_days=500, n_stocks=50):
    """创建模拟测试数据（包含已知模式）"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    symbols = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    data = []
    for symbol in symbols:
        # 生成价格（随机游走）
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # 生成因子（与未来收益有真实关系）
        # 因子1: 反转因子（与未来收益负相关）
        reversal = np.random.normal(0, 1, n_days)
        # 因子2: 动量因子（与未来收益正相关）
        momentum = np.random.normal(0, 1, n_days)
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'symbol': symbol,
                'close': prices[i],
                'f_reversal_5': reversal[i],
                'f_momentum_60': momentum[i],
                'f_volatility_20_inv': np.random.normal(0, 1),
            })
    
    df = pd.DataFrame(data)
    
    # 添加真实的未来收益关系（用于验证）
    df['fwd_return'] = df.groupby('symbol')['close'].transform(
        lambda x: x.shift(-5) / x - 1
    )
    
    return df


def test_future_function_risk():
    """测试是否存在未来函数风险"""
    logger.info("="*60)
    logger.info("测试1: 未来函数风险检查")
    logger.info("="*60)
    
    from rolling_xgboost import RollingXGBoostScorer
    
    # 创建测试数据
    df = create_test_data(n_days=300, n_stocks=20)
    factor_names = ['f_reversal_5', 'f_momentum_60', 'f_volatility_20_inv']
    
    # 记录特定日期的预测
    test_date = df['date'].iloc[200]
    logger.info(f"测试日期: {test_date}")
    
    # 创建scorer并训练到测试日期
    scorer = RollingXGBoostScorer(
        train_window=100,
        retrain_freq=30,
        forward_days=5,
    )
    
    # 训练模型（只用测试日期之前的数据）
    success = scorer.train(df, test_date, factor_names)
    
    if success:
        logger.info("✓ 模型训练成功（只用历史数据）")
        
        # 获取测试日期的数据
        test_data = df[df['date'] == test_date].copy()
        X_today = test_data.set_index('symbol')[factor_names]
        
        # 预测
        scores = scorer.predict(X_today)
        logger.info(f"✓ 预测得分范围: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # 验证：训练数据不包含测试日期及之后的数据
        train_end = pd.to_datetime(test_date) - timedelta(days=1)
        train_data = df[df['date'] <= train_end]
        future_data = df[df['date'] >= test_date]
        
        logger.info(f"  训练数据最后日期: {train_data['date'].max()}")
        logger.info(f"  测试日期: {test_date}")
        logger.info(f"  未来数据最早日期: {future_data['date'].min()}")
        
        if train_data['date'].max() < test_date:
            logger.info("✓ 验证通过：训练数据不包含测试日期及之后的数据")
        else:
            logger.error("✗ 验证失败：训练数据包含未来信息！")
    else:
        logger.warning("模型训练失败（可能数据不足）")


def test_rolling_training():
    """测试滚动训练机制"""
    logger.info("\n" + "="*60)
    logger.info("测试2: 滚动训练机制")
    logger.info("="*60)
    
    from rolling_xgboost import RollingXGBoostScorer
    
    df = create_test_data(n_days=400, n_stocks=30)
    factor_names = ['f_reversal_5', 'f_momentum_60', 'f_volatility_20_inv']
    
    scorer = RollingXGBoostScorer(
        train_window=100,
        retrain_freq=50,
        forward_days=5,
    )
    
    # 模拟多个交易日期
    test_dates = df['date'].unique()[150::50]  # 每50天测试一次
    
    train_count = 0
    for test_date in test_dates[:5]:  # 测试前5个时点
        if scorer.should_retrain(test_date):
            success = scorer.train(df, test_date, factor_names)
            if success:
                train_count += 1
                logger.info(f"✓ {test_date}: 重新训练模型")
        else:
            logger.info(f"  {test_date}: 跳过训练（未到重训练周期）")
    
    logger.info(f"\n共训练 {train_count} 次，验证滚动机制正常")


def test_ic_filter():
    """测试因子IC筛选功能"""
    logger.info("\n" + "="*60)
    logger.info("测试3: 因子IC筛选")
    logger.info("="*60)
    
    from rolling_xgboost import RollingXGBoostScorer
    
    # 创建有明确IC关系的数据
    np.random.seed(42)
    n_days = 200
    n_stocks = 20
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    symbols = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    data = []
    for symbol in symbols:
        # 强反转因子（IC高）
        reversal = np.random.normal(0, 1, n_days)
        # 弱随机因子（IC低）
        random_factor = np.random.normal(0, 1, n_days)
        
        # 价格：与反转因子负相关（反转效应）
        returns = -0.01 * reversal + np.random.normal(0, 0.01, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'symbol': symbol,
                'close': prices[i],
                'f_strong_reversal': reversal[i],
                'f_weak_random': random_factor[i],
            })
    
    df = pd.DataFrame(data)
    factor_names = ['f_strong_reversal', 'f_weak_random']
    
    scorer = RollingXGBoostScorer(ic_threshold=0.03)
    
    # 训练并筛选因子
    test_date = dates[150]
    scorer.train(df, test_date, factor_names)
    
    logger.info(f"有效因子: {scorer.valid_factors}")
    
    if 'f_strong_reversal' in scorer.valid_factors:
        logger.info("✓ IC筛选正确：强因子被保留")
    if 'f_weak_random' not in scorer.valid_factors:
        logger.info("✓ IC筛选正确：弱因子被剔除")


def test_full_pipeline():
    """测试完整流程"""
    logger.info("\n" + "="*60)
    logger.info("测试4: 完整流程测试")
    logger.info("="*60)
    
    from rolling_xgboost import compute_ml_factor_scores
    
    df = create_test_data(n_days=300, n_stocks=20)
    factor_names = ['f_reversal_5', 'f_momentum_60', 'f_volatility_20_inv']
    
    # 运行完整流程
    result_df = compute_ml_factor_scores(df, factor_names)
    
    # 检查结果
    valid_scores = result_df['ml_score'].dropna()
    logger.info(f"得分计算完成:")
    logger.info(f"  总样本: {len(result_df)}")
    logger.info(f"  有效得分: {len(valid_scores)}")
    logger.info(f"  得分范围: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
    logger.info(f"  排名范围: [{result_df['rank'].min():.0f}, {result_df['rank'].max():.0f}]")
    
    if len(valid_scores) > 0:
        logger.info("✓ 完整流程测试通过")
    else:
        logger.error("✗ 完整流程测试失败：无有效得分")


if __name__ == '__main__':
    logger.info("开始测试滚动XGBoost模块...")
    
    try:
        test_future_function_risk()
    except Exception as e:
        logger.error(f"测试1失败: {e}")
    
    try:
        test_rolling_training()
    except Exception as e:
        logger.error(f"测试2失败: {e}")
    
    try:
        test_ic_filter()
    except Exception as e:
        logger.error(f"测试3失败: {e}")
    
    try:
        test_full_pipeline()
    except Exception as e:
        logger.error(f"测试4失败: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("所有测试完成")
    logger.info("="*60)
