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
from rolling_xgboost import RollingXGBoostScorer
from rolling_ic_weight_fast import RollingICWeighterFast as RollingICWeighter
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_factor_scores(df, use_ml=False, use_rolling_ml=False, use_rolling_ic=False):
    """
    计算每个交易日每只股票的综合因子得分。

    权重分配方式：
      --fast / 默认: 固定线性加权（Config.FACTOR_WEIGHTS）
      --use-rolling-ic: 滚动IC加权（IC数据驱动权重，60日滚动窗口）
      --use-rolling-ml: 滚动XGBoost机器学习（全因子输入，自动学习权重）

    所有选股结果通过 SELECTION_MAP 传递给执行层，
    执行层直接买入，不加任何额外过滤器。
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

    # ===== Step 3: 计算综合得分（固定线性加权） =====
    df['composite_score'] = 0.0
    
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
        
        # 只选择FACTOR_WEIGHTS中定义的优质因子，过滤低IR因子
        ic_factor_names = [f for f in factor_names 
                          if f.replace('f_', '') in Config.FACTOR_WEIGHTS]
        logger.info(f"  滚动IC使用 {len(ic_factor_names)}/{len(factor_names)} 个优质因子")
        removed_factors = set(factor_names) - set(ic_factor_names)
        if removed_factors:
            logger.info(f"  排除的因子: {[f.replace('f_', '') for f in removed_factors]}")
        
        try:
            weighter = RollingICWeighter(
                lookback=60,           # 60日IC回看
                update_freq=15,        # 每15日更新权重（接近典型信号半衰期）
                forward_days=5,        # 预测5日收益
                min_ic=0.01,           # IC阈值
                decay_factor=0.9,      # 时间衰减
                max_weight=0.30,       # 单因子最大30%
            )
            df = weighter.compute_scores(df, ic_factor_names)
            
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
        logger.info("  注意：前60个交易日将使用线性加权（积累足够训练数据）")
        
        # XGBoost使用全部因子，让模型自己学习特征重要性
        # 不同于线性加权需要筛选因子，XGBoost能自动处理特征冗余和重要性排序
        ml_factor_names = factor_names[:]  # 全部24个因子
        logger.info(f"  滚动XGBoost使用全部 {len(ml_factor_names)} 个因子")
        
        try:
            # 根据数据量自动调整参数
            n_samples = len(df)
            n_stocks = df['symbol'].nunique()
            
            # 小样本时降低要求
            min_train_samples = max(100, n_stocks * 10)
            
            logger.info(f"  数据量: {n_samples}条, {n_stocks}只股票")
            
            scorer = RollingXGBoostScorer(
                train_window=252,          # 1年训练窗口（更稳定）
                retrain_freq=42,           # 每2个月重训练
                forward_days=5,
                ic_threshold=0.015,        # 只保留IC绝对值>0.015的因子
                min_train_samples=min_train_samples,
                n_estimators=50,           # 减少树数量，防过拟合
                max_depth=3,               # 浅层树，防止学到虚假规律
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=5.0,            # 更强的L2正则化
                min_child_weight=10,       # 叶子节点最少样本数
            )
            df = scorer.compute_scores(df, ml_factor_names)
            
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

    # ===== Step 6: 自动计算信号半衰期并匹配参数 =====
    half_life = _compute_signal_half_life(df)
    Config.MIN_HOLD_BARS = half_life
    Config.SELL_THRESHOLD = Config.TOP_N_STOCKS * max(2, half_life // 3 + 1)
    print(f"  ✓ 信号半衰期 ≈ {half_life} 天 → MIN_HOLD_BARS 已设为 {half_life}")
    print(f"  ✓ SELL_THRESHOLD 已动态设为 TOP_N × {max(2, half_life // 3 + 1)} = {Config.SELL_THRESHOLD}")
    logger.info(f"  ✓ MIN_HOLD_BARS 已动态设为 {half_life}（信号半衰期）")
    logger.info(f"  ✓ SELL_THRESHOLD 已动态设为 TOP_N × {max(2, half_life // 3 + 1)} = {Config.SELL_THRESHOLD}")

    # ===== Step 7: 全局排名 =====
    # 因子评估显示得分越高的股票收益越高（Q1 > Q10），所以使用降序排名
    # ascending=False → 得分最高的股票排在最前面（rank=1）
    df['rank'] = df.groupby('date')['composite_score'].rank(ascending=False, method='first')

    # ===== Step 8: 选股信号 =====
    df['selected'] = np.where(df['rank'] <= Config.SELL_THRESHOLD, 1, 0)

    logger.info(f"因子得分计算完成，每日选取前 {Config.TOP_N_STOCKS} 只")
    return df


def _compute_signal_half_life(df):
    """计算综合得分的信号半衰期（自相关跌破 0.5 的天数）。"""
    all_acf = []
    for sym, grp in df.groupby('symbol'):
        scores = grp['composite_score'].dropna().values
        if len(scores) > 20:
            for lag in range(1, min(21, len(scores))):
                if len(scores) > lag:
                    acf = np.corrcoef(scores[:-lag], scores[lag:])[0, 1]
                    if not np.isnan(acf):
                        all_acf.append((lag, acf))
    acf_df = pd.DataFrame(all_acf, columns=['lag', 'acf'])

    half_life = 15  # 默认值
    for lag in range(1, 21):
        vals = acf_df[acf_df['lag'] == lag]['acf']
        if len(vals) > 0 and vals.mean() < 0.5:
            half_life = lag
            break

    print(f"\n[自动匹配] 综合得分信号半衰期 ≈ {half_life} 天")
    logger.info(f"\n[自动匹配] 综合得分信号半衰期 ≈ {half_life} 天")
    return half_life


SELECTION_MAP = {}

def build_daily_selections(df):
    """从因子打分结果构建每日选股映射表。（按排名排序）"""
    selected_df = df[df['selected'] == 1][['date', 'symbol', 'rank']].copy()
    selected_df['date'] = pd.to_datetime(selected_df['date'])
    selected_df = selected_df.sort_values(['date', 'rank'])
    selection_map = {}
    for date, group in selected_df.groupby('date'):
        selection_map[date] = group['symbol'].tolist()  # 有序列表，排名1~150
    print(f"  ✓ 构建每日选股表：{len(selection_map)} 个交易日")
    return selection_map
