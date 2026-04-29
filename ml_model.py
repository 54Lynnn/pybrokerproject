# -*- coding: utf-8 -*-
"""XGBoost机器学习模型 — 替代线性加权"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def prepare_training_data(df, factor_names, forward_days=5):
    """
    准备训练数据：用当日因子预测未来N日收益。
    
    注意：避免未来函数！
    - X = 当日因子值（t日收盘后计算）
    - y = 未来N日收益（t+1到t+N日）
    """
    print(f"\n[ML] 正在准备训练数据...")
    
    # 计算未来收益（作为标签）
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    df['fwd_close'] = df.groupby('symbol')['close'].shift(-forward_days)
    df['fwd_return'] = df['fwd_close'] / df['close'] - 1
    
    # 删除未来数据无法计算的行
    df = df.dropna(subset=['fwd_return'])
    
    # 特征 = 因子值
    X = df[factor_names].values
    y = df['fwd_return'].values
    
    # 按日期分组，确保时间序列顺序
    dates = df['date'].values
    symbols = df['symbol'].values
    
    print(f"  ✓ 训练样本数: {len(X)}")
    print(f"  ✓ 特征维度: {X.shape[1]}")
    print(f"  ✓ 目标变量: 未来{forward_days}日收益率")
    
    return X, y, dates, symbols, df


def train_xgboost_model(X, y, dates, n_splits=5):
    """
    用时间序列交叉验证训练XGBoost模型。
    
    注意：必须用时间序列分割，避免未来信息泄露！
    """
    print(f"\n[ML] 正在训练XGBoost模型...")
    
    # 按日期排序
    sort_idx = np.argsort(dates)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    models = []
    train_scores = []
    val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
        X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
        y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]
        
        # 训练XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 评估
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        
        models.append(model)
        train_scores.append(train_r2)
        val_scores.append(val_r2)
        
        print(f"  Fold {fold+1}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
    
    # 选择验证集表现最好的模型
    best_fold = np.argmax(val_scores)
    best_model = models[best_fold]
    
    print(f"\n  ✓ 最佳模型: Fold {best_fold+1}")
    print(f"  ✓ 训练集 R²: {train_scores[best_fold]:.4f}")
    print(f"  ✓ 验证集 R²: {val_scores[best_fold]:.4f}")
    
    # 特征重要性
    feature_importance = best_model.feature_importances_
    
    return best_model, feature_importance


def predict_scores(model, X):
    """
    用训练好的模型预测得分。
    """
    scores = model.predict(X)
    return scores


def ml_factor_scoring(df, factor_names):
    """
    主函数：用XGBoost替代线性加权计算综合得分。
    """
    print(f"\n{'='*60}")
    print(f"  XGBoost机器学习因子打分")
    print(f"{'='*60}")
    
    # 准备数据
    X, y, dates, symbols, df_processed = prepare_training_data(df, factor_names)
    
    # 训练模型
    model, importance = train_xgboost_model(X, y, dates)
    
    # 用模型预测所有样本的得分
    print(f"\n[ML] 正在用模型预测得分...")
    scores = predict_scores(model, X)
    
    # 将得分添加回DataFrame
    df_processed['ml_score'] = scores
    
    # 按日期排名
    df_processed['rank'] = df_processed.groupby('date')['ml_score'].rank(ascending=False, method='first')
    
    print(f"  ✓ ML得分计算完成")
    print(f"  ✓ 得分范围: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return df_processed, model, importance
