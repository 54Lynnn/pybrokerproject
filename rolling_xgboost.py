# -*- coding: utf-8 -*-
"""
滚动XGBoost排序学习因子打分器 — 严格无未来函数

核心改进：
  使用XGBRanker（排序学习）替代XGBRegressor（回归）。
  
  为什么排序学习更适合选股：
    - 回归：预测"未来5日涨2.3%"，精确预测涨幅几乎不可能
    - 排序：预测"这只在1800只里排第45"，只关心相对好坏
    - 排序对噪声更鲁棒，天然匹配"选前N只"的选股逻辑

  实现原理：
    - 每个交易日作为一个query group
    - group内按未来N日收益排序生成标签
    - 模型学习如何让好股票排在坏股票前面
    - 使用NDCG作为优化目标（信息检索领域标准排序指标）

使用方法：
  from rolling_xgboost import RollingXGBRanker
  scorer = RollingXGBRanker()
  df = scorer.compute_scores(df, factor_names)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class RollingXGBRanker:
    """
    滚动XGBoost排序学习因子打分器。
    
    严格避免未来函数：
      - 训练集严格在预测时点之前
      - 模型定期重新训练，不跨期使用
      - 特征和标签都严格按时间对齐
    
    排序学习优势（相对回归）：
      - 只关心相对排序，不care具体涨跌幅
      - 对噪声因子更鲁棒（弱因子只是影响小，不会破坏模型）
      - 天然匹配"选前N只"的选股目标
    """
    
    def __init__(
        self,
        train_window=252,        # 训练窗口（交易日，约1年）
        retrain_freq=63,         # 重训练频率（交易日，约3个月）
        forward_days=5,          # 预测未来N日收益
        min_train_samples=10000, # 最小训练样本数
        # XGBRanker参数
        n_estimators=100,        # 100棵决策树
        max_depth=4,             # 深度4（排序比回归抗过拟合，可稍深）
        learning_rate=0.1,
        subsample=0.8,           # 行采样80%
        colsample_bytree=0.8,    # 列采样80%
        reg_alpha=0.5,           # L1正则化（适度）
        reg_lambda=2.0,          # L2正则化（适度，不如回归那么严）
        min_child_weight=5,
        random_state=42
    ):
        self.train_window = train_window
        self.retrain_freq = retrain_freq
        self.forward_days = forward_days
        self.min_train_samples = min_train_samples
        
        # XGBRanker参数
        self.xgb_params = {
            'objective': 'rank:ndcg',      # 排序学习目标：NDCG
            'eval_metric': 'ndcg@10',      # 评估指标：前10名NDCG
            'ndcg_exp_gain': False,        # 使用线性增益（标签值可>31，兼容每日1800只股票）
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        # 状态变量
        self.model = None
        self.last_train_date = None
        self.factor_names = None
        
        logger.info("RollingXGBRanker 初始化完成")
        logger.info(f"  训练窗口: {train_window}日")
        logger.info(f"  重训练频率: {retrain_freq}日")
        logger.info(f"  预测目标: 未来{forward_days}日收益排序")
        logger.info(f"  排序算法: XGBRanker (objective=rank:ndcg)")
    
    def _prepare_ranking_data(self, df, factor_names):
        """
        准备排序学习训练数据。
        
        核心逻辑：
          1. 计算每只股票未来N日收益率
          2. 在每一天内，按收益率排序生成标签
          3. 记录每天有多少只股票（group sizes）
        
        标签含义：
          label = 1, 2, 3, ..., N  (N = 当天股票数)
          数值越大 = 未来收益越高 = 应该排在前面
        
        Args:
            df: 包含因子值和收盘价的DataFrame
            factor_names: 因子列名列表
        
        Returns:
            X: 特征矩阵
            y: 排序标签（整数，越大越好）
            group_sizes: 每天股票数的数组
        """
        df_temp = df[['date', 'symbol', 'close'] + factor_names].copy()
        
        # 计算未来N日收益率
        df_temp['fwd_return'] = df_temp.groupby('symbol')['close'].transform(
            lambda x: x.shift(-self.forward_days) / x - 1
        )
        
        # 删除无法计算未来收益的行
        df_temp = df_temp.dropna(subset=factor_names + ['fwd_return'])
        
        if df_temp.empty:
            return None, None, None
        
        # 按日期排序，确保group顺序一致
        df_temp = df_temp.sort_values('date').reset_index(drop=True)
        
        # 分位数标签：按未来收益分为10组（1~10）
        # 用分位数代替原始排名更稳健：
        #   - 原始排名：每天有1800级标签，模型过度关注极端值
        #   - 分位数：每天只有10级，模型学"好/中/差"的区分，更抗过拟合
        df_temp['label'] = df_temp.groupby('date')['fwd_return'].transform(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop') + 1
        )
        
        X = df_temp[factor_names].values
        y = df_temp['label'].values.astype(int)
        
        # group sizes: 每天有多少只股票
        group_sizes = df_temp.groupby('date').size().values
        
        return X, y, group_sizes, df_temp[['date', 'symbol']]
    
    def train(self, df, current_date, factor_names):
        """
        训练排序模型（严格只用历史数据）。
        
        Args:
            df: 完整数据
            current_date: 当前预测日期（训练集必须在此之前）
            factor_names: 因子列名列表
        
        Returns:
            bool: 训练是否成功
        """
        logger.info(f"\n[ML] 正在训练排序模型 (当前日期: {current_date})...")
        
        # 1. 提取训练窗口数据（严格在current_date之前）
        train_end = pd.to_datetime(current_date) - timedelta(days=1)
        
        available_history = df[df['date'] < current_date]['date'].unique()
        available_days = len(available_history)
        logger.info(f"  可用历史数据: {available_days}天")
        
        if available_days < self.train_window:
            adjusted_window = max(60, available_days - 10)
            logger.warning(f"  历史数据不足{self.train_window}天，调整训练窗口为{adjusted_window}天")
            train_window_days = adjusted_window
        else:
            train_window_days = self.train_window
        
        train_start = train_end - timedelta(days=train_window_days * 2)
        
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        train_df = df.loc[train_mask].copy()
        
        if len(train_df) < self.min_train_samples:
            logger.warning(f"  训练样本不足: {len(train_df)} < {self.min_train_samples}，跳过训练")
            return False
        
        logger.info(f"  训练集: {train_start.date()} ~ {train_end.date()}, {len(train_df)}条")
        
        # 2. 准备排序学习数据
        X, y, group_sizes, _ = self._prepare_ranking_data(train_df, factor_names)
        
        if X is None or len(X) < self.min_train_samples:
            logger.warning(f"  有效训练样本不足，跳过训练")
            return False
        
        if len(group_sizes) < 10:
            logger.warning(f"  交易日数量不足: {len(group_sizes)}天，至少需要10天")
            return False
        
        # 3. 训练XGBRanker
        logger.info(f"  训练样本: {len(X)}, 特征维度: {X.shape[1]}, 交易日: {len(group_sizes)}天")
        logger.info(f"  每日股票数: min={group_sizes.min()}, max={group_sizes.max()}, mean={group_sizes.mean():.0f}")
        
        self.model = xgb.XGBRanker(**self.xgb_params)
        self.model.fit(X, y, group=group_sizes)
        
        # 4. 记录特征重要性
        self.factor_names = factor_names
        importance = dict(zip(factor_names, self.model.feature_importances_))
        logger.info(f"  特征重要性 (Top 10):")
        for factor, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"    {factor}: {imp:.4f}")
        
        # 5. 评估训练集NDCG
        y_pred = self.model.predict(X)
        train_ndcg = _compute_ndcg(y, y_pred, group_sizes, k=10)
        logger.info(f"  训练集NDCG@10: {train_ndcg:.4f}")
        
        self.last_train_date = pd.to_datetime(current_date)
        logger.info(f"  ✓ 排序模型训练完成")
        
        return True
    
    def should_retrain(self, current_date):
        """判断是否需要重新训练。"""
        if self.model is None:
            return True
        if self.last_train_date is None:
            return True
        days_diff = (pd.to_datetime(current_date) - self.last_train_date).days
        return days_diff >= self.retrain_freq
    
    def predict(self, X_today):
        """
        预测排序得分（越高越好）。
        
        XGBRanker输出的分值可以直接作为排序依据，
        分值越高表示模型认为该股票越应该排在前面。
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()")
        scores = self.model.predict(X_today)
        return scores
    
    def compute_scores(self, df, factor_names):
        """
        主函数：滚动训练XGBRanker并计算因子得分。
        
        严格按时间顺序处理，避免任何未来函数。
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  滚动XGBoost排序学习因子打分")
        logger.info(f"{'='*60}")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        trading_dates = sorted(df['date'].unique())
        logger.info(f"  总交易日: {len(trading_dates)}")
        
        df['ml_score'] = np.nan
        
        train_count = 0
        predict_count = 0
        
        for i, current_date in enumerate(trading_dates):
            if i < self.train_window:
                continue
            
            if self.should_retrain(current_date):
                success = self.train(df, current_date, factor_names)
                if success:
                    train_count += 1
            
            if self.model is not None:
                day_mask = df['date'] == current_date
                day_df = df.loc[day_mask].copy()
                
                if len(day_df) > 0:
                    try:
                        X_today = day_df[factor_names].values
                        scores = self.model.predict(X_today)
                        df.loc[day_mask, 'ml_score'] = scores
                        predict_count += len(day_df)
                    except Exception as e:
                        logger.warning(f"  预测失败 {current_date}: {e}")
                        continue
            
            if i % 63 == 0:
                logger.info(f"  进度: {i}/{len(trading_dates)} 交易日, 已训练{train_count}次")
        
        valid_scores = df['ml_score'].dropna()
        logger.info(f"\n  滚动训练完成:")
        logger.info(f"    训练次数: {train_count}")
        logger.info(f"    预测样本: {len(valid_scores)}")
        logger.info(f"    得分范围: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
        logger.info(f"    得分均值: {valid_scores.mean():.4f}")
        
        return df


def _compute_ndcg(y_true, y_pred, group_sizes, k=10):
    """
    计算NDCG@K（归一化折损累计增益）。
    
    用于评估排序质量：
      - 完美的排序 NDCG=1.0
      - 随机排序 NDCG≈0.0
    
    Args:
        y_true: 真实标签（越大越好）
        y_pred: 预测得分
        group_sizes: 每天股票数的数组
        k: 只评估前k名
    
    Returns:
        float: NDCG@K 均值
    """
    ndcg_scores = []
    start = 0
    for size in group_sizes:
        end = start + size
        true = y_true[start:end]
        pred = y_pred[start:end]
        
        if len(true) <= 1:
            start = end
            continue
        
        # 按预测排序
        order = np.argsort(pred)[::-1]
        true_sorted = true[order]
        
        # 取前k个
        true_k = true_sorted[:min(k, len(true))]
        
        # DCG
        dcg = true_k[0]
        for j in range(1, len(true_k)):
            dcg += true_k[j] / np.log2(j + 2)
        
        # IDCG（理想排序）
        ideal_order = np.argsort(true)[::-1]
        ideal_sorted = true[ideal_order][:min(k, len(true))]
        idcg = ideal_sorted[0]
        for j in range(1, len(ideal_sorted)):
            idcg += ideal_sorted[j] / np.log2(j + 2)
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        
        start = end
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0
