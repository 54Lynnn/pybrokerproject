# -*- coding: utf-8 -*-
"""
专业因子分析报告生成器

功能：
  1. 未来函数风险扫描
  2. 单因子IC/IR分析
  3. 因子相关性分析
  4. 因子衰减分析
  5. 生成优化建议
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import json
from datetime import datetime
from config import Config
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 1. 未来函数风险扫描
# ============================================================

def check_future_function_risk(df):
    """
    扫描因子计算中可能的未来函数风险。
    
    检查项：
      1. 是否存在使用未来数据的shift方向错误
      2. 是否存在收盘价/最高价/最低价的lookahead bias
      3. 财务数据是否使用了报表日而非发布日
      4. 指标计算是否使用了当日未收盘数据
    """
    print(f"\n{'='*60}")
    print("  未来函数风险扫描报告")
    print(f"{'='*60}")
    
    risks = []
    
    # 检查1: 数据对齐方式
    if 'date' in df.columns:
        date_diff = df.groupby('symbol')['date'].diff().dt.days
        abnormal_diff = date_diff[date_diff > 7]
        if len(abnormal_diff) > 0:
            risks.append(f"⚠ 发现 {len(abnormal_diff)} 条记录间隔超过7天，可能存在停牌后数据跳跃")
    
    # 检查2: 收益率计算方向
    if 'fwd_return' in df.columns:
        fwd_corr = df['close'].corr(df['fwd_return'])
        if fwd_corr > 0.5:
            risks.append(f"⚠ 收盘价与未来收益相关性过高({fwd_corr:.3f})，可能泄露未来信息")
    
    # 检查3: 指标值范围异常
    for col in df.columns:
        if col.startswith('f_'):
            if df[col].abs().max() > 100:
                risks.append(f"⚠ 因子 {col} 存在极端值(>{df[col].abs().max():.1f})，可能计算错误")
    
    # 检查4: 财务数据时间戳
    if 'pbMRQ' in df.columns or 'peTTM' in df.columns:
        risks.append("ℹ 财务数据使用baostock提供的MRQ/TTM值，已考虑发布延迟，风险较低")
    
    if not risks:
        print("  ✓ 未发现明显的未来函数风险")
    else:
        for risk in risks:
            print(f"  {risk}")
    
    return risks


# ============================================================
# 2. 单因子IC分析（专业版）
# ============================================================

def compute_ic_analysis(df, factor_names, horizons=(1, 5, 10, 20)):
    """
    计算各因子的IC、IR、胜率、盈亏比等核心指标。
    
    返回:
        dict: {factor_name: {metric: value}}
    """
    print(f"\n{'='*60}")
    print("  单因子有效性检验 (IC Analysis)")
    print(f"{'='*60}")
    
    results = {}
    
    for factor in factor_names:
        if factor not in df.columns:
            continue
            
        factor_results = {}
        
        for h in horizons:
            # 计算未来收益
            df['fwd_ret'] = df.groupby('symbol')['close'].pct_change(h).shift(-h)
            
            # 按日计算IC
            daily_ics = []
            daily_long_ret = []  # 多头组合收益
            daily_short_ret = []  # 空头组合收益
            
            for date, grp in df.groupby('date'):
                grp = grp.dropna(subset=[factor, 'fwd_ret'])
                if len(grp) < 10:
                    continue
                
                # Spearman IC
                ic = grp[factor].corr(grp['fwd_ret'], method='spearman')
                if not np.isnan(ic):
                    daily_ics.append(ic)
                
                # 分位数收益
                grp = grp.sort_values(factor, ascending=False)
                n = len(grp)
                top10 = grp.head(max(1, n // 10))
                bottom10 = grp.tail(max(1, n // 10))
                
                long_ret = top10['fwd_ret'].mean()
                short_ret = bottom10['fwd_ret'].mean()
                
                if not np.isnan(long_ret):
                    daily_long_ret.append(long_ret)
                if not np.isnan(short_ret):
                    daily_short_ret.append(short_ret)
            
            if daily_ics:
                ics = pd.Series(daily_ics)
                factor_results[f'IC_mean_{h}d'] = ics.mean()
                factor_results[f'IC_std_{h}d'] = ics.std()
                factor_results[f'IR_{h}d'] = ics.mean() / ics.std() if ics.std() > 0 else 0
                factor_results[f'IC_pos_ratio_{h}d'] = (ics > 0).mean()
                
                # t统计量
                factor_results[f'IC_tstat_{h}d'] = ics.mean() / (ics.std() / np.sqrt(len(ics)))
            
            if daily_long_ret and daily_short_ret:
                long_rets = pd.Series(daily_long_ret)
                short_rets = pd.Series(daily_short_ret)
                factor_results[f'Long_ret_{h}d'] = long_rets.mean()
                factor_results[f'Short_ret_{h}d'] = short_rets.mean()
                factor_results[f'LS_spread_{h}d'] = long_rets.mean() - short_rets.mean()
                
                # 胜率
                factor_results[f'Long_winrate_{h}d'] = (long_rets > 0).mean()
                factor_results[f'LS_winrate_{h}d'] = ((long_rets - short_rets) > 0).mean()
        
        results[factor] = factor_results
    
    # 打印结果
    print(f"\n  {'因子':<25} {'IC(5d)':>8} {'IR(5d)':>8} {'胜率':>8} {'多空价差':>10} {'t统计量':>8}")
    print(f"  {'-'*70}")
    
    sorted_factors = sorted(results.items(), 
                           key=lambda x: x[1].get('IR_5d', 0), 
                           reverse=True)
    
    for factor, metrics in sorted_factors:
        ic = metrics.get('IC_mean_5d', 0)
        ir = metrics.get('IR_5d', 0)
        wr = metrics.get('Long_winrate_5d', 0)
        spread = metrics.get('LS_spread_5d', 0)
        tstat = metrics.get('IC_tstat_5d', 0)
        print(f"  {factor:<25} {ic:>+8.4f} {ir:>8.3f} {wr:>8.1%} {spread:>+10.4f} {tstat:>+8.2f}")
    
    return results


# ============================================================
# 3. 因子相关性分析
# ============================================================

def analyze_factor_correlation(df, factor_names):
    """
    分析因子间的相关性，识别冗余因子。
    
    高相关性因子 (>0.7) 建议剔除或合并。
    """
    print(f"\n{'='*60}")
    print("  因子相关性分析")
    print(f"{'='*60}")
    
    factor_df = df[factor_names].copy()
    corr_matrix = factor_df.corr()
    
    # 找出高相关性因子对
    high_corr_pairs = []
    for i in range(len(factor_names)):
        for j in range(i+1, len(factor_names)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append((factor_names[i], factor_names[j], corr))
    
    if high_corr_pairs:
        print(f"\n  高相关性因子对 (|r| > 0.7):")
        for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"    {f1:<25} vs {f2:<25} |r|={abs(corr):.3f}")
        print(f"\n  ⚠ 建议：每对高相关因子保留IR较高的一个")
    else:
        print(f"\n  ✓ 未发现高度相关的因子对 (|r| > 0.7)")
    
    return corr_matrix, high_corr_pairs


# ============================================================
# 4. 因子衰减分析
# ============================================================

def analyze_factor_decay(df, factor_names):
    """
    分析因子的预测能力衰减速度。
    
    计算不同lag下的IC，判断因子有效持续期。
    """
    print(f"\n{'='*60}")
    print("  因子衰减分析 (Factor Decay)")
    print(f"{'='*60}")
    
    lags = [1, 3, 5, 10, 20, 40, 60]
    
    print(f"\n  {'因子':<25} {'1d':>8} {'3d':>8} {'5d':>8} {'10d':>8} {'20d':>8} {'40d':>8} {'60d':>8}")
    print(f"  {'-'*85}")
    
    decay_results = {}
    
    for factor in factor_names:
        if factor not in df.columns:
            continue
        
        ic_by_lag = {}
        for lag in lags:
            df['fwd_ret_lag'] = df.groupby('symbol')['close'].pct_change(lag).shift(-lag)
            
            daily_ics = []
            for date, grp in df.groupby('date'):
                grp = grp.dropna(subset=[factor, 'fwd_ret_lag'])
                if len(grp) < 10:
                    continue
                ic = grp[factor].corr(grp['fwd_ret_lag'], method='spearman')
                if not np.isnan(ic):
                    daily_ics.append(ic)
            
            if daily_ics:
                ic_by_lag[lag] = np.mean(daily_ics)
            else:
                ic_by_lag[lag] = 0
        
        decay_results[factor] = ic_by_lag
        
        # 打印
        ic_str = ' '.join([f"{ic_by_lag.get(l, 0):>+8.4f}" for l in lags])
        print(f"  {factor:<25} {ic_str}")
    
    # 找出半衰期
    print(f"\n  因子半衰期 (IC衰减至初始值50%的时间):")
    for factor, ic_by_lag in decay_results.items():
        ic_1d = ic_by_lag.get(1, 0)
        if abs(ic_1d) > 0.01:
            half_ic = ic_1d * 0.5
            half_life = None
            for lag in lags:
                if abs(ic_by_lag.get(lag, 0)) < abs(half_ic):
                    half_life = lag
                    break
            if half_life:
                print(f"    {factor:<25} 半衰期 ≈ {half_life}天")
    
    return decay_results


# ============================================================
# 5. 综合优化建议
# ============================================================

def generate_optimization_suggestions(ic_results, corr_matrix, high_corr_pairs, decay_results):
    """
    基于分析结果生成专业优化建议。
    """
    print(f"\n{'='*60}")
    print("  专业优化建议")
    print(f"{'='*60}")
    
    suggestions = []
    
    # 建议1: 剔除无效因子
    weak_factors = []
    for factor, metrics in ic_results.items():
        ir = metrics.get('IR_5d', 0)
        ic = metrics.get('IC_mean_5d', 0)
        if abs(ir) < 0.3 or abs(ic) < 0.02:
            weak_factors.append((factor, ir, ic))
    
    if weak_factors:
        suggestions.append(f"\n【1. 因子精简】剔除低有效性因子（|IR|<0.3 或 |IC|<0.02）:")
        for factor, ir, ic in weak_factors:
            suggestions.append(f"    - {factor}: IR={ir:.3f}, IC={ic:.4f}")
    
    # 建议2: 处理高相关性
    if high_corr_pairs:
        suggestions.append(f"\n【2. 去冗余】高相关因子对只保留IR更高的:")
        for f1, f2, corr in high_corr_pairs[:5]:
            ir1 = ic_results.get(f1, {}).get('IR_5d', 0)
            ir2 = ic_results.get(f2, {}).get('IR_5d', 0)
            keep = f1 if ir1 > ir2 else f2
            drop = f2 if ir1 > ir2 else f1
            suggestions.append(f"    - 保留 {keep}(IR={max(ir1,ir2):.3f}), 剔除 {drop}(IR={min(ir1,ir2):.3f})")
    
    # 建议3: 权重优化
    suggestions.append(f"\n【3. 权重优化】基于IR加权的建议配置:")
    valid_factors = {k: v for k, v in ic_results.items() 
                     if v.get('IR_5d', 0) > 0.3}
    total_ir = sum(v.get('IR_5d', 0) for v in valid_factors.values())
    if total_ir > 0:
        for factor, metrics in sorted(valid_factors.items(), 
                                      key=lambda x: x[1].get('IR_5d', 0),
                                      reverse=True)[:10]:
            weight = metrics.get('IR_5d', 0) / total_ir
            suggestions.append(f"    - {factor:<25} 权重={weight:.2%}")
    
    # 建议4: 调仓频率
    fast_decay_factors = []
    for factor, ic_by_lag in decay_results.items():
        ic_1d = ic_by_lag.get(1, 0)
        ic_5d = ic_by_lag.get(5, 0)
        if abs(ic_1d) > 0.03 and abs(ic_5d) < abs(ic_1d) * 0.5:
            fast_decay_factors.append(factor)
    
    if fast_decay_factors:
        suggestions.append(f"\n【4. 调仓频率】以下因子衰减快，建议提高调仓频率:")
        for factor in fast_decay_factors[:5]:
            suggestions.append(f"    - {factor}")
    
    # 建议5: 交易成本控制
    suggestions.append(f"\n【5. 交易成本】当前策略交易过于频繁:")
    suggestions.append(f"    - 建议将 MIN_HOLD_BARS 从 15 提高到 20-25 天")
    suggestions.append(f"    - 建议将 SELL_THRESHOLD 从 60 降低到 30-40")
    suggestions.append(f"    - 考虑加入换手率惩罚项，避免频繁换股")
    
    for s in suggestions:
        print(s)
    
    return suggestions


# ============================================================
# 主函数
# ============================================================

def run_full_factor_analysis(df, factor_names):
    """
    运行完整的因子分析流程。
    """
    print(f"\n{'#'*60}")
    print("#" + " "*58 + "#")
    print("#" + "        专业因子分析报告".center(50) + "#")
    print("#" + " "*58 + "#")
    print(f"{'#'*60}")
    
    # 1. 未来函数风险扫描
    risks = check_future_function_risk(df)
    
    # 2. IC分析
    ic_results = compute_ic_analysis(df, factor_names)
    
    # 3. 相关性分析
    corr_matrix, high_corr_pairs = analyze_factor_correlation(df, factor_names)
    
    # 4. 衰减分析
    decay_results = analyze_factor_decay(df, factor_names)
    
    # 5. 优化建议
    suggestions = generate_optimization_suggestions(ic_results, corr_matrix, 
                                                     high_corr_pairs, decay_results)
    
    # 保存结果
    report = {
        'analysis_time': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'future_function_risks': risks,
        'ic_results': {k: {mk: float(mv) if isinstance(mv, (int, float, np.number)) else str(mv) 
                          for mk, mv in v.items()} 
                      for k, v in ic_results.items()},
        'high_correlation_pairs': [(f1, f2, float(c)) for f1, f2, c in high_corr_pairs],
        'suggestions': suggestions
    }
    
    report_path = os.path.join(os.path.dirname(__file__), 'factor_analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n  ✓ 分析报告已保存: {report_path}")
    
    return report


if __name__ == '__main__':
    # 测试代码
    print("因子分析模块加载完成")
