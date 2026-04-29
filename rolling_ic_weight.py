# -*- coding: utf-8 -*-
"""
滚动IC加权模块 — 无未来函数的自适应权重系统

核心设计：
  1. 滚动计算每个因子的IC（只用历史数据）
  2. 基于历史IC动态调整权重
  3. IC高的因子给高权重，IC低/负的因子给低权重或剔除
  4. 严格避免未来函数：只用当前时点之前的数据计算IC

使用方法：
  from rolling_ic_weight import RollingICWeighter
  weighter = RollingICWeighter(lookback=60, min_ic=0.01)
  df = weighter.compute_weights(df, factor_names)
"""

import numpy as np
import pandas as pd
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class RollingICWeighter:
    """
    滚动IC加权器。
    
    严格无未来函数：
      - 每次计算IC只用当前日期之前的数据
      - 权重基于历史IC的衰减加权平均
      - 定期更新权重（如每月）
    """
    
    def __init__(
        self,
        lookback=60,           # IC计算回看窗口（交易日）
        update_freq=20,        # 权重更新频率（交易日，约1个月）
        forward_days=5,        # 预测未来N日收益
        min_ic=0.01,           # 最小IC阈值（|IC|<此值的因子权重为0）
        decay_factor=0.9,      # 时间衰减因子（近期IC权重更高）
        max_weight=0.30,       # 单个因子最大权重
        min_weight=0.0,        # 单个因子最小权重
    ):
        self.lookback = lookback
        self.update_freq = update_freq
        self.forward_days = forward_days
        self.min_ic = min_ic
        self.decay_factor = decay_factor
        self.max_weight = max_weight
        self.min_weight = min_weight
        
        # 状态变量
        self.current_weights = None
        self.last_update_date = None
        self.ic_history = {}   # 存储IC历史 {factor: [ic1, ic2, ...]}
        
        logger.info("RollingICWeighter 初始化完成")
        logger.info(f"  IC回看窗口: {lookback}日")
        logger.info(f"  权重更新频率: {update_freq}日")
        logger.info(f"  预测目标: 未来{forward_days}日收益")
        logger.info(f"  IC阈值: {min_ic}")
        logger.info(f"  时间衰减: {decay_factor}")
    
    def _calculate_ic(self, df, factor_col, current_date):
        """
        计算单因子在回看窗口内的IC。
        
        严格无未来函数：
          - 只用 current_date 之前 lookback 天的数据
          - 因子值是历史数据，收益是未来N日（但在历史区间内）
        """
        # 提取回看窗口数据（严格在current_date之前）
        window_start = pd.to_datetime(current_date) - timedelta(days=self.lookback * 2)
        window_mask = (df['date'] >= window_start) & (df['date'] < current_date)
        window_df = df.loc[window_mask].copy()
        
        if len(window_df) < 100:  # 数据不足
            return 0.0
        
        # 计算未来收益（标签）
        window_df['fwd_return'] = window_df.groupby('symbol')['close'].transform(
            lambda x: x.shift(-self.forward_days) / x - 1
        )
        
        # 删除缺失值
        valid_data = window_df.dropna(subset=[factor_col, 'fwd_return'])
        
        if len(valid_data) < 100:
            return 0.0
        
        # 按日计算Rank IC
        daily_ic = []
        for date, group in valid_data.groupby('date'):
            if len(group) < 5:
                continue
            try:
                ic = group[factor_col].corr(group['fwd_return'], method='spearman')
                if not np.isnan(ic) and not np.isinf(ic):
                    daily_ic.append(ic)
            except:
                continue
        
        if len(daily_ic) < 10:
            return 0.0
        
        # 返回平均IC
        return np.mean(daily_ic)
    
    def _update_weights(self, df, current_date, factor_names):
        """
        更新因子权重（基于历史IC）。
        
        严格无未来函数：只用 current_date 之前的数据。
        """
        logger.info(f"\n[IC加权] 更新权重 (日期: {current_date})...")
        
        # 计算每个因子的IC
        ic_values = {}
        for factor in factor_names:
            ic = self._calculate_ic(df, factor, current_date)
            ic_values[factor] = ic
            
            # 更新IC历史
            if factor not in self.ic_history:
                self.ic_history[factor] = []
            self.ic_history[factor].append(ic)
            
            # 只保留最近20次IC记录
            if len(self.ic_history[factor]) > 20:
                self.ic_history[factor] = self.ic_history[factor][-20:]
        
        # 显示IC值
        logger.info(f"  因子IC值:")
        for factor, ic in sorted(ic_values.items(), key=lambda x: abs(x[1]), reverse=True):
            status = "✓" if abs(ic) >= self.min_ic else "✗"
            logger.info(f"    {status} {factor}: IC={ic:+.4f}")
        
        # 基于IC计算权重
        weights = {}
        for factor in factor_names:
            ic = ic_values[factor]
            
            # IC绝对值低于阈值的因子权重为0
            if abs(ic) < self.min_ic:
                weights[factor] = 0.0
                continue
            
            # 权重 = |IC| * sign(IC)  # 保持方向
            # 但我们要的是正向预测能力，所以用IC的绝对值作为权重基础
            weight_base = abs(ic)
            
            weights[factor] = weight_base
        
        # 归一化权重（使总和为1）
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # 如果所有因子IC都太低，使用等权
            weights = {k: 1.0 / len(factor_names) for k in factor_names}
        
        # 限制最大/最小权重
        for factor in weights:
            weights[factor] = np.clip(weights[factor], self.min_weight, self.max_weight)
        
        # 再次归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        self.current_weights = weights
        self.last_update_date = pd.to_datetime(current_date)
        
        logger.info(f"  更新后权重:")
        for factor, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if w > 0.001:
                logger.info(f"    {factor}: {w:.4f} ({w*100:.1f}%)")
        
        return weights
    
    def should_update(self, current_date):
        """判断是否需要更新权重。"""
        if self.current_weights is None:
            return True
        if self.last_update_date is None:
            return True
        
        days_diff = (pd.to_datetime(current_date) - self.last_update_date).days
        return days_diff >= self.update_freq
    
    def compute_scores(self, df, factor_names):
        """
        主函数：滚动IC加权计算因子得分。
        
        严格按时间顺序处理，避免任何未来函数。
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  滚动IC加权因子打分")
        logger.info(f"{'='*60}")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # 获取所有交易日期
        trading_dates = sorted(df['date'].unique())
        logger.info(f"  总交易日: {len(trading_dates)}")
        
        # 初始化得分列
        df['ic_weighted_score'] = np.nan
        
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
            
            # 获取当日数据
            day_mask = df['date'] == current_date
            day_df = df.loc[day_mask].copy()
            
            if len(day_df) > 0 and self.current_weights is not None:
                # 计算加权得分
                scores = np.zeros(len(day_df))
                for factor, weight in self.current_weights.items():
                    if factor in day_df.columns and weight > 0:
                        scores += day_df[factor].fillna(0).values * weight
                
                # 回填得分
                df.loc[day_mask, 'ic_weighted_score'] = scores
            
            # 每月输出进度
            if i % 20 == 0:
                logger.info(f"  进度: {i}/{len(trading_dates)} 交易日, 已更新{update_count}次权重")
        
        # 统计结果
        valid_scores = df['ic_weighted_score'].dropna()
        logger.info(f"\n  滚动IC加权完成:")
        logger.info(f"    权重更新次数: {update_count}")
        logger.info(f"    有效得分样本: {len(valid_scores)}")
        logger.info(f"    得分范围: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
        
        # 计算排名
        df['rank'] = df.groupby('date')['ic_weighted_score'].rank(ascending=False, method='first')
        
        return df


def compute_ic_weighted_scores(df, factor_names, **kwargs):
    """
    便捷函数：用滚动IC加权计算因子得分。
    
    参数:
        df: 包含因子列的DataFrame
        factor_names: 因子列名列表
        **kwargs: 传递给RollingICWeighter的参数
    
    返回:
        df: 添加了'ic_weighted_score'和'rank'列的DataFrame
    """
    weighter = RollingICWeighter(**kwargs)
    return weighter.compute_scores(df, factor_names)
