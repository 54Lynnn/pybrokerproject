# -*- coding: utf-8 -*-
"""
滚动XGBoost训练模块 — 严格无未来函数实现

核心设计原则：
  1. 训练集严格在预测时点之前
  2. 定期重新训练，丢弃旧模型
  3. 强正则化防止过拟合
  4. 特征筛选（只保留IC显著的因子）
  5. 滚动窗口自适应市场变化

使用方法：
  from rolling_xgboost import RollingXGBoostScorer
  scorer = RollingXGBoostScorer(train_window=252, retrain_freq=63)
  df = scorer.compute_scores(df, factor_names)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class RollingXGBoostScorer:
    """
    滚动XGBoost因子打分器。
    
    严格避免未来函数：
      - 每次训练只用当前时点之前的数据
      - 模型定期更新，不跨期使用
      - 特征和标签都严格按时间对齐
    """
    
    def __init__(
        self,
        train_window=252,        # 训练窗口（交易日，约1年）
        retrain_freq=63,         # 重训练频率（交易日，约3个月）
        forward_days=5,          # 预测未来N日收益
        min_train_samples=500,   # 最小训练样本数
        ic_threshold=0.02,       # 因子IC筛选阈值
        # XGBoost参数（强正则化）
        n_estimators=50,         # 减少树数量
        max_depth=3,             # 浅层树
        learning_rate=0.05,
        subsample=0.7,           # 行采样
        colsample_bytree=0.7,    # 列采样
        reg_alpha=1.0,           # L1正则化
        reg_lambda=2.0,          # L2正则化
        min_child_weight=10,     # 最小叶子样本数
        random_state=42
    ):
        self.train_window = train_window
        self.retrain_freq = retrain_freq
        self.forward_days = forward_days
        self.min_train_samples = min_train_samples
        self.ic_threshold = ic_threshold
        
        # XGBoost参数
        self.xgb_params = {
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
        }
        
        # 状态变量
        self.model = None
        self.last_train_date = None
        self.valid_factors = None  # 经过IC筛选的有效因子
        self.factor_importance = None
        
        logger.info("RollingXGBoostScorer 初始化完成")
        logger.info(f"  训练窗口: {train_window}日")
        logger.info(f"  重训练频率: {retrain_freq}日")
        logger.info(f"  预测目标: 未来{forward_days}日收益")
        logger.info(f"  IC筛选阈值: {ic_threshold}")
    
    def _calculate_ic(self, df, factor_col, forward_days=5):
        """
        计算单因子的IC（信息系数）。
        
        严格无未来函数：
          - 因子值用当日数据
          - 收益用未来N日数据（shift(-N)）
        """
        df_temp = df[['date', 'symbol', factor_col, 'close']].copy()
        
        # 计算未来收益（标签）
        df_temp['fwd_return'] = df_temp.groupby('symbol')['close'].transform(
            lambda x: x.shift(-forward_days) / x - 1
        )
        
        # 删除无法计算未来收益的行
        df_temp = df_temp.dropna(subset=[factor_col, 'fwd_return'])
        
        if len(df_temp) < 30:
            return 0.0
        
        # 按日计算Rank IC
        daily_ic = []
        for date, group in df_temp.groupby('date'):
            if len(group) < 5:
                continue
            try:
                ic = group[factor_col].corr(group['fwd_return'], method='spearman')
                if not np.isnan(ic):
                    daily_ic.append(ic)
            except:
                continue
        
        return np.mean(daily_ic) if daily_ic else 0.0
    
    def _filter_factors(self, df, factor_names):
        """
        筛选有效因子：只保留|IC| > 阈值的因子。
        
        避免引入未来函数：
          - IC计算只在训练集内进行
          - 筛选后的因子列表在当前窗口固定
        """
        logger.info("  正在进行因子IC筛选...")
        
        valid_factors = []
        ic_results = {}
        
        for factor in factor_names:
            ic = self._calculate_ic(df, factor, self.forward_days)
            ic_results[factor] = ic
            
            if abs(ic) >= self.ic_threshold:
                valid_factors.append(factor)
                logger.info(f"    ✓ {factor}: IC={ic:.4f} (保留)")
            else:
                logger.info(f"    ✗ {factor}: IC={ic:.4f} (剔除)")
        
        # 如果没有因子通过筛选，保留IC绝对值最大的3个
        if len(valid_factors) < 3:
            sorted_factors = sorted(ic_results.items(), key=lambda x: abs(x[1]), reverse=True)
            valid_factors = [f for f, _ in sorted_factors[:3]]
            logger.warning(f"  因子IC均较低，强制保留前3个因子")
        
        logger.info(f"  因子筛选完成: {len(valid_factors)}/{len(factor_names)} 个因子有效")
        return valid_factors, ic_results
    
    def _prepare_training_data(self, df, factor_names):
        """
        准备训练数据。
        
        严格无未来函数：
          - X = 当日因子值（历史数据计算）
          - y = 未来N日收益（仅作为标签，不参与特征计算）
        """
        df_temp = df[['date', 'symbol', 'close'] + factor_names].copy()
        
        # 计算未来收益（标签）
        df_temp['fwd_return'] = df_temp.groupby('symbol')['close'].transform(
            lambda x: x.shift(-self.forward_days) / x - 1
        )
        
        # 删除缺失值
        df_temp = df_temp.dropna(subset=factor_names + ['fwd_return'])
        
        X = df_temp[factor_names].values
        y = df_temp['fwd_return'].values
        dates = df_temp['date'].values
        
        return X, y, dates, df_temp
    
    def train(self, df, current_date, factor_names):
        """
        在当前时点训练模型（严格只用历史数据）。
        
        参数:
            df: 完整数据（包含历史和未来，但训练时只取历史部分）
            current_date: 当前预测日期（训练集必须在此之前）
            factor_names: 因子名称列表
        """
        logger.info(f"\n[ML] 正在训练模型 (当前日期: {current_date})...")
        
        # 1. 提取训练窗口数据（严格在current_date之前）
        train_end = pd.to_datetime(current_date) - timedelta(days=1)
        
        # 计算实际可用的历史数据天数
        available_history = df[df['date'] < current_date]['date'].unique()
        available_days = len(available_history)
        
        logger.info(f"  可用历史数据: {available_days}天")
        
        # 如果历史数据不足，自适应调整训练窗口
        if available_days < self.train_window:
            adjusted_window = max(60, available_days - 10)  # 留10天缓冲
            logger.warning(f"  历史数据不足{self.train_window}天，调整训练窗口为{adjusted_window}天")
            train_window_days = adjusted_window
        else:
            train_window_days = self.train_window
        
        train_start = train_end - timedelta(days=train_window_days * 2)  # 留足交易日
        
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        train_df = df.loc[train_mask].copy()
        
        if len(train_df) < self.min_train_samples:
            logger.warning(f"  训练样本不足: {len(train_df)} < {self.min_train_samples}，跳过训练")
            return False
        
        logger.info(f"  训练集: {train_start.date()} ~ {train_end.date()}, {len(train_df)}条")
        
        # 2. 因子IC筛选（只在训练集内进行）
        self.valid_factors, ic_results = self._filter_factors(train_df, factor_names)
        
        # 3. 准备训练数据
        X, y, dates, _ = self._prepare_training_data(train_df, self.valid_factors)
        
        if len(X) < self.min_train_samples:
            logger.warning(f"  有效训练样本不足: {len(X)} < {self.min_train_samples}")
            return False
        
        # 4. 训练XGBoost模型
        logger.info(f"  训练样本: {len(X)}, 特征维度: {X.shape[1]}")
        
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X, y)
        
        # 5. 记录特征重要性
        self.factor_importance = dict(zip(self.valid_factors, self.model.feature_importances_))
        logger.info(f"  特征重要性:")
        for factor, importance in sorted(self.factor_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {factor}: {importance:.4f}")
        
        # 6. 评估训练集表现
        y_pred = self.model.predict(X)
        train_ic = np.corrcoef(y_pred, y)[0, 1]
        logger.info(f"  训练集IC: {train_ic:.4f}")
        
        self.last_train_date = pd.to_datetime(current_date)
        logger.info(f"  ✓ 模型训练完成")
        
        return True
    
    def should_retrain(self, current_date):
        """判断是否需要重新训练模型。"""
        if self.model is None:
            return True
        if self.last_train_date is None:
            return True
        
        days_diff = (pd.to_datetime(current_date) - self.last_train_date).days
        return days_diff >= self.retrain_freq
    
    def predict(self, X_today):
        """
        用当前模型预测得分。
        
        注意：模型只包含历史信息，没有未来函数。
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()")
        
        # 只使用训练时的有效因子
        X = X_today[self.valid_factors].values
        scores = self.model.predict(X)
        
        return scores
    
    def compute_scores(self, df, factor_names):
        """
        主函数：滚动训练XGBoost并计算因子得分。
        
        严格按时间顺序处理，避免任何未来函数。
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  滚动XGBoost因子打分")
        logger.info(f"{'='*60}")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # 获取所有交易日期
        trading_dates = sorted(df['date'].unique())
        logger.info(f"  总交易日: {len(trading_dates)}")
        
        # 初始化得分列
        df['ml_score'] = np.nan
        
        # 滚动训练和预测
        train_count = 0
        predict_count = 0
        
        for i, current_date in enumerate(trading_dates):
            # 跳过前期数据不足的阶段
            if i < self.train_window:
                continue
            
            # 检查是否需要重新训练
            if self.should_retrain(current_date):
                success = self.train(df, current_date, factor_names)
                if success:
                    train_count += 1
            
            # 如果模型已训练，进行预测
            if self.model is not None:
                # 获取当日数据
                day_mask = df['date'] == current_date
                day_df = df.loc[day_mask].copy()
                
                if len(day_df) > 0:
                    try:
                        # 预测得分（只用历史训练的模型）
                        X_today = day_df.set_index('symbol')[self.valid_factors]
                        scores = self.predict(X_today)
                        
                        # 回填得分
                        df.loc[day_mask, 'ml_score'] = scores
                        predict_count += len(day_df)
                    except Exception as e:
                        logger.warning(f"  预测失败 {current_date}: {e}")
                        continue
            
            # 每季度输出进度
            if i % 63 == 0:
                logger.info(f"  进度: {i}/{len(trading_dates)} 交易日, 已训练{train_count}次")
        
        # 统计结果
        valid_scores = df['ml_score'].dropna()
        logger.info(f"\n  滚动训练完成:")
        logger.info(f"    训练次数: {train_count}")
        logger.info(f"    预测样本: {len(valid_scores)}")
        logger.info(f"    得分范围: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
        logger.info(f"    得分均值: {valid_scores.mean():.4f}")
        
        # 计算排名
        df['rank'] = df.groupby('date')['ml_score'].rank(ascending=False, method='first')
        
        return df


def compute_ml_factor_scores(df, factor_names, **kwargs):
    """
    便捷函数：用滚动XGBoost计算因子得分。
    
    参数:
        df: 包含因子列的DataFrame
        factor_names: 因子列名列表（如 ['f_reversal_5', 'f_momentum_60', ...]）
        **kwargs: 传递给RollingXGBoostScorer的参数
    
    返回:
        df: 添加了'ml_score'和'rank'列的DataFrame
    """
    scorer = RollingXGBoostScorer(**kwargs)
    return scorer.compute_scores(df, factor_names)
