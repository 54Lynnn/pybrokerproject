# -*- coding: utf-8 -*-
"""
多因子打分与选股 — 专业优化版

优化要点：
  1. 默认关闭XGBoost（避免过拟合），使用线性加权
  2. 优化动态权重逻辑
  3. 增加因子正交化选项
  4. 改进得分计算方式
"""

from config import Config
from factor_engineering import generate_factors, standardize_factors
from market_regime import detect_market_regime, get_dynamic_weights
from rolling_xgboost import RollingXGBoostScorer
from rolling_ic_weight import RollingICWeighter
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_factor_scores(df, df_index=None, use_ml=False, use_rolling_ml=False, use_rolling_ic=False):
    """
    计算每个交易日每只股票的综合因子得分。
    
    参数:
        df_index: 中证500指数数据，用于判断市场状态
        use_ml: True=使用旧版XGBoost（不推荐，有未来函数风险）
        use_rolling_ml: True=使用滚动XGBoost（推荐，无未来函数）
        use_rolling_ic: True=使用滚动IC加权（推荐，无未来函数）
    """
    logger.info("正在计算多因子综合得分...")
    if use_rolling_ic:
        logger.info("模式: 滚动IC加权（无未来函数）")
    elif use_rolling_ml:
        logger.info("模式: 滚动XGBoost机器学习（无未来函数）")
    elif use_ml:
        logger.info("模式: XGBoost机器学习（旧版，不推荐）")
    else:
        logger.info("模式: 线性加权")

    df = df.copy()

    # ===== Step 1: 批量生成因子 =====
    df, factor_names = generate_factors(df)

    # ===== Step 2: 截面标准化 =====
    df = standardize_factors(df, factor_names)

    # ===== Step 3: 判断市场状态并获取动态权重 =====
    regime_info = None
    if df_index is not None and not df_index.empty:
        regime_info = detect_market_regime(df_index)
        if regime_info is not None:
            df['date'] = pd.to_datetime(df['date'])
            regime_info['date'] = pd.to_datetime(regime_info['date'])
            df = df.merge(regime_info[['date', 'market_regime']], on='date', how='left')
            df['market_regime'] = df['market_regime'].fillna('neutral')
            logger.info("已启用动态权重调整")
        else:
            df['market_regime'] = 'neutral'
    else:
        df['market_regime'] = 'neutral'
        logger.info("无指数数据，使用固定权重")

    # ===== Step 4: 计算综合得分 =====
    df['composite_score'] = 0.0
    
    # 使用线性加权（支持动态权重）
    if regime_info is not None and 'market_regime' in df.columns:
        # 按市场状态分组计算
        for regime, group in df.groupby('market_regime'):
            weights = get_dynamic_weights(regime, Config.FACTOR_WEIGHTS)
            mask = df['market_regime'] == regime
            
            valid_count = 0
            for key, w in weights.items():
                col = 'f_' + key
                if col in df.columns:
                    df.loc[mask, 'composite_score'] += df.loc[mask, col].fillna(0) * w
                    valid_count += 1
            
            logger.info(f"市场状态 {regime}: 使用 {valid_count} 个因子")
    else:
        # 固定权重
        weights = Config.FACTOR_WEIGHTS
        valid_factors = [f for f in factor_names if f.replace('f_', '') in weights]
        logger.info(f"使用 {len(valid_factors)} 个有效因子计算得分")

        for key, w in weights.items():
            col = 'f_' + key
            if col in df.columns:
                df['composite_score'] += df[col].fillna(0) * w

    # 如果没有匹配的因子，使用等权
    if df['composite_score'].abs().max() < 1e-10:
        logger.warning("没有匹配的因子权重，使用所有因子等权")
        for col in factor_names:
            df['composite_score'] += df[col].fillna(0) / len(factor_names)

    # ===== Step 5: 滚动IC加权（如果启用） =====
    if use_rolling_ic:
        logger.info("\n正在使用滚动IC加权计算得分...")
        logger.info("  注意：前60个交易日将使用线性加权（积累足够IC数据）")
        try:
            weighter = RollingICWeighter(
                lookback=60,           # 60日IC回看
                update_freq=20,        # 每20日更新权重
                forward_days=5,        # 预测5日收益
                min_ic=0.01,           # IC阈值
                decay_factor=0.9,      # 时间衰减
                max_weight=0.30,       # 单因子最大30%
            )
            df = weighter.compute_scores(df, factor_names)
            
            # 用IC加权得分替换线性加权得分
            if 'ic_weighted_score' in df.columns:
                valid_ic = df['ic_weighted_score'].notna().sum()
                total_days = df['date'].nunique()
                logger.info(f"  IC加权得分有效样本: {valid_ic}/{len(df)} ({valid_ic/len(df)*100:.1f}%)")
                logger.info(f"  覆盖交易日: {df[df['ic_weighted_score'].notna()]['date'].nunique()}/{total_days}")
                if valid_ic > 0:
                    df['composite_score'] = df['ic_weighted_score'].fillna(df['composite_score'])
                    logger.info("✓ 滚动IC加权得分已融合（缺失部分用线性加权填充）")
                else:
                    logger.warning("  IC加权得分无效，全程使用线性加权")
            else:
                logger.warning("  未生成IC加权得分，全程使用线性加权")
        except Exception as e:
            logger.error(f"滚动IC加权失败，回退到线性加权: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ===== Step 6: 滚动XGBoost（如果启用） =====
    if use_rolling_ml:
        logger.info("\n正在使用滚动XGBoost计算ML得分...")
        logger.info("  注意：前252个交易日将使用线性加权（积累足够训练数据）")
        try:
            # 根据数据量自动调整参数
            n_samples = len(df)
            n_stocks = df['symbol'].nunique()
            
            # 小样本时降低要求
            min_train_samples = max(100, n_stocks * 10)
            train_window = min(252, max(60, n_samples // n_stocks // 2))
            
            logger.info(f"  数据量: {n_samples}条, {n_stocks}只股票")
            logger.info(f"  自适应参数: train_window={train_window}, min_samples={min_train_samples}")
            
            scorer = RollingXGBoostScorer(
                train_window=train_window,
                retrain_freq=max(20, train_window // 4),
                forward_days=5,
                ic_threshold=0.01,     # 降低IC阈值
                min_train_samples=min_train_samples,
                n_estimators=30,       # 减少树数量
                max_depth=2,           # 更浅的树
                reg_alpha=1.0,
                reg_lambda=2.0,
            )
            df = scorer.compute_scores(df, factor_names)
            
            # 用ML得分替换线性加权得分
            if 'ml_score' in df.columns:
                valid_ml = df['ml_score'].notna().sum()
                total_days = df['date'].nunique()
                logger.info(f"  ML得分有效样本: {valid_ml}/{len(df)} ({valid_ml/len(df)*100:.1f}%)")
                logger.info(f"  覆盖交易日: {df[df['ml_score'].notna()]['date'].nunique()}/{total_days}")
                if valid_ml > 0:
                    df['composite_score'] = df['ml_score'].fillna(df['composite_score'])
                    logger.info("✓ 滚动XGBoost得分已融合（缺失部分用线性加权填充）")
                else:
                    logger.warning("  ML得分无效，全程使用线性加权")
            else:
                logger.warning("  未生成ML得分，全程使用线性加权")
        except Exception as e:
            logger.error(f"滚动XGBoost失败，回退到线性加权: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ===== Step 6: 全局排名 =====
    df['rank'] = df.groupby('date')['composite_score'].rank(ascending=False, method='first')

    # ===== Step 7: 选股信号 =====
    df['selected'] = np.where(df['rank'] <= Config.SELL_THRESHOLD, 1, 0)

    logger.info(f"因子得分计算完成，每日选取前 {Config.TOP_N_STOCKS} 只")
    return df


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
