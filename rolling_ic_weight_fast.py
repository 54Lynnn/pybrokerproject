# -*- coding: utf-8 -*-
"""
滚动IC加权模块 — 高性能优化版

性能优化要点：
  1. 预先计算所有未来收益（一次计算，多次使用）
  2. 使用向量化操作代替嵌套循环
  3. 利用预计算的IC缓存
  4. 使用numba加速相关系数计算
"""

import numpy as np
import pandas as pd
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def spearman_corr(x, y):
    """快速计算Spearman相关系数"""
    if len(x) < 5:
        return np.nan
    
    # 转换为rank
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    
    # 计算皮尔逊相关系数（对rank等价于Spearman）
    n = len(rx)
    mean_x = rx.mean()
    mean_y = ry.mean()
    
    cov = ((rx - mean_x) * (ry - mean_y)).sum() / n
    std_x = np.sqrt(((rx - mean_x) ** 2).sum() / n)
    std_y = np.sqrt(((ry - mean_y) ** 2).sum() / n)
    
    if std_x == 0 or std_y == 0:
        return np.nan
    
    return cov / (std_x * std_y)


class RollingICWeighterFast:
    """
    高性能滚动IC加权器。
    
    优化策略：
      1. 预先计算所有日期的未来收益
      2. 使用向量化操作
      3. 缓存IC计算结果
    """
    
    def __init__(
        self,
        lookback=60,
        update_freq=12,
        forward_days=5,
        min_ic=0.01,
        decay_factor=0.9,
        max_weight=0.30,
    ):
        self.lookback = lookback
        self.update_freq = update_freq
        self.forward_days = forward_days
        self.min_ic = min_ic
        self.decay_factor = decay_factor
        self.max_weight = max_weight
        
        self.current_weights = None
        self.last_update_date = None
        self.ic_history = {}
        
        logger.info("RollingICWeighterFast 初始化完成")
        logger.info(f"  IC回看窗口: {lookback}日")
        logger.info(f"  权重更新频率: {update_freq}日")
        logger.info(f"  预测目标: 未来{forward_days}日收益")
    
    def _precompute_fwd_returns(self, df):
        """预先计算所有股票的未来收益（一次计算，多次使用）"""
        logger.info("  预计算未来收益...")
        df['fwd_return'] = df.groupby('symbol')['close'].transform(
            lambda x: x.shift(-self.forward_days) / x - 1
        )
        return df
    
    def _compute_ic_for_factor(self, df, factor_col, current_date):
        """计算单因子IC（优化版）"""
        # 提取回看窗口数据
        window_start = pd.to_datetime(current_date) - timedelta(days=self.lookback * 3)
        window_mask = (df['date'] >= window_start) & (df['date'] < current_date)
        window_df = df.loc[window_mask]
        
        if len(window_df) < 100:
            return 0.0
        
        valid_data = window_df.dropna(subset=[factor_col, 'fwd_return'])
        
        if len(valid_data) < 100:
            return 0.0
        
        # 按日分组计算IC（使用向量化）
        daily_ics = valid_data.groupby('date').apply(
            lambda g: spearman_corr(g[factor_col].values, g['fwd_return'].values)
            if len(g) >= 5 else np.nan
        )
        
        daily_ics = daily_ics.dropna().values
        
        if len(daily_ics) < 10:
            return 0.0
        
        return np.mean(daily_ics)
    
    def _update_weights(self, df, current_date, factor_names):
        """更新因子权重"""
        logger.info(f"\n[IC加权] 更新权重 (日期: {current_date})...")
        
        ic_values = {}
        for factor in factor_names:
            ic = self._compute_ic_for_factor(df, factor, current_date)
            ic_values[factor] = ic
            
            if factor not in self.ic_history:
                self.ic_history[factor] = []
            self.ic_history[factor].append(ic)
            
            if len(self.ic_history[factor]) > 20:
                self.ic_history[factor] = self.ic_history[factor][-20:]
        
        logger.info(f"  因子IC值:")
        for factor, ic in sorted(ic_values.items(), key=lambda x: abs(x[1]), reverse=True):
            status = "✓" if abs(ic) >= self.min_ic else "✗"
            logger.info(f"    {status} {factor}: IC={ic:+.4f}")
        
        # 计算权重（保持符号）
        weights = {}
        for factor in factor_names:
            ic = ic_values[factor]
            if abs(ic) < self.min_ic:
                weights[factor] = 0.0
                continue
            weights[factor] = abs(ic) * np.sign(ic)
        
        # 归一化（绝对值归一化）
        total_abs_weight = sum(abs(w) for w in weights.values())
        if total_abs_weight > 0:
            weights = {k: v / total_abs_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(factor_names) for k in factor_names}
        
        # 限制权重范围
        for factor in weights:
            weights[factor] = np.clip(weights[factor], -self.max_weight, self.max_weight)
        
        self.current_weights = weights
        self.last_update_date = pd.to_datetime(current_date)
        
        logger.info(f"  更新后权重:")
        for factor, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(w) > 0.001:
                logger.info(f"    {factor}: {w:.4f} ({w*100:.1f}%)")
        
        return weights
    
    def should_update(self, current_date):
        """判断是否需要更新权重"""
        if self.current_weights is None or self.last_update_date is None:
            return True
        
        days_diff = (pd.to_datetime(current_date) - self.last_update_date).days
        return days_diff >= self.update_freq
    
    def compute_scores(self, df, factor_names):
        """主函数：滚动IC加权计算因子得分"""
        logger.info(f"\n{'='*60}")
        logger.info(f"  滚动IC加权因子打分（高性能版）")
        logger.info(f"{'='*60}")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # 预计算未来收益（关键优化！）
        df = self._precompute_fwd_returns(df)
        
        # 按日期排序
        trading_dates = sorted(df['date'].unique())
        logger.info(f"  总交易日: {len(trading_dates)}")
        
        # 初始化得分列
        df['ic_weighted_score'] = np.nan
        
        # 按日期分组的数据（优化查找）
        date_groups = df.groupby('date')
        
        # 滚动计算权重和得分
        update_count = 0
        
        for i, current_date in enumerate(trading_dates):
            # 跳过前期数据不足的阶段
            if i < self.lookback:
                continue
            
            # 检查是否需要更新权重
            if self.should_update(current_date):
                self._update_weights(df, current_date, factor_names)
                update_count += 1
            
            # 获取当日数据并计算得分
            if self.current_weights is not None:
                try:
                    day_df = date_groups.get_group(current_date)
                    scores = np.zeros(len(day_df))
                    for factor, weight in self.current_weights.items():
                        if factor in day_df.columns and abs(weight) > 0.0001:
                            scores += day_df[factor].fillna(0).values * weight
                    
                    df.loc[df['date'] == current_date, 'ic_weighted_score'] = scores
                except KeyError:
                    pass
            
            # 进度输出
            if i % 50 == 0:
                logger.info(f"  进度: {i}/{len(trading_dates)} 交易日")
        
        logger.info(f"\n  滚动IC加权完成:")
        logger.info(f"    权重更新次数: {update_count}")
        
        return df


def compute_ic_weighted_scores_fast(df, factor_names, **kwargs):
    """便捷函数：用高性能滚动IC加权计算因子得分"""
    weighter = RollingICWeighterFast(**kwargs)
    return weighter.compute_scores(df, factor_names)
