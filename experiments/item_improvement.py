#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的PHQ-9子项目回归评估
只计算各个子项目的MAE和RMSE，不涉及复杂的聚合方法
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from item_predictor import ItemPredictor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_phq9_items():
    """评估PHQ-9各个子项目的预测性能"""
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    acoustic_path = os.path.join(project_root, 'dataset', 'CS-NRAC-E.csv')
    questionnaire_path = os.path.join(project_root, 'dataset', 'raw_info.csv')
    
    logger.info("开始数据处理...")
    
    # 数据处理
    data_processor = DataProcessor(acoustic_path, questionnaire_path)
    data_dict = data_processor.process()
    
    # PHQ-9项目列表
    phq9_items = ['PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9']
    phq9_available = [item for item in phq9_items if item in data_dict['Y_train'].columns]
    
    logger.info(f"可用的PHQ-9项目: {phq9_available}")
    
    # 训练项目级预测器
    logger.info("训练PHQ-9项目级预测器...")
    item_predictor = ItemPredictor(
        data_dict['X_train'],
        data_dict['Y_train'],
        data_dict['X_val'],
        data_dict['Y_val'],
        data_dict['X_test']
    )
    
    # 训练模型
    item_predictor.train_models(items=phq9_available, n_jobs=10)
    
    # 生成测试集预测
    Y_test_pred = item_predictor.predict(items=phq9_available, n_jobs=10)
    
    logger.info("开始评估各项目性能...")
    
    # 计算各项目的MAE和RMSE
    results = {}
    
    print("\n" + "="*80)
    print("PHQ-9各子项目回归性能评估")
    print("="*80)
    print(f"{'项目':<8} {'真实均值':<10} {'预测均值':<10} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
    print("-"*80)
    
    mae_sum = 0
    rmse_sum = 0
    
    for item in phq9_available:
        # 真实值和预测值
        true_values = data_dict['Y_test'][item].values
        pred_values = Y_test_pred[item].values
        
        # 计算指标
        mae = mean_absolute_error(true_values, pred_values)
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        r2 = r2_score(true_values, pred_values)
        
        mae_sum += mae
        rmse_sum += rmse
        
        # 保存结果
        results[item] = {
            'true_mean': np.mean(true_values),
            'pred_mean': np.mean(pred_values),
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        # 打印结果
        print(f"{item:<8} {results[item]['true_mean']:<10.2f} {results[item]['pred_mean']:<10.2f} "
              f"{mae:<8.3f} {rmse:<8.3f} {r2:<8.3f}")
    
    print("-"*80)
    print(f"{'平均':<8} {'':<10} {'':<10} {mae_sum/len(phq9_available):<8.3f} {rmse_sum/len(phq9_available):<8.3f}")
    print(f"{'总和':<8} {'':<10} {'':<10} {mae_sum:<8.3f} {rmse_sum:<8.3f}")
    
    # 计算总分对比（简单加和）
    print("\n" + "="*60)
    print("总分预测对比（简单加和）")
    print("="*60)
    
    true_total = data_dict['Y_test'][phq9_available].sum(axis=1).values
    pred_total = Y_test_pred.sum(axis=1).values
    
    total_mae = mean_absolute_error(true_total, pred_total)
    total_rmse = np.sqrt(mean_squared_error(true_total, pred_total))
    total_r2 = r2_score(true_total, pred_total)
    
    print(f"总分MAE: {total_mae:.3f}")
    print(f"总分RMSE: {total_rmse:.3f}")
    print(f"总分R²: {total_r2:.3f}")
    print(f"真实总分范围: {true_total.min():.1f} - {true_total.max():.1f}")
    print(f"预测总分范围: {pred_total.min():.1f} - {pred_total.max():.1f}")
    
    print("\n解释MAE差异:")
    print(f"各项目MAE之和: {mae_sum:.3f}")
    print(f"总分MAE: {total_mae:.3f}")
    print(f"差异: {mae_sum - total_mae:.3f}")
    print("这是正常现象：|e1| + |e2| + ... ≠ |e1 + e2 + ...|")
    
    # 保存结果
    output_dir = './results/simple_phq9_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, 'phq9_item_performance.csv'))
    
    # 保存预测对比
    comparison_df = pd.DataFrame({
        'true_total': true_total,
        'pred_total': pred_total
    })
    for item in phq9_available:
        comparison_df[f'true_{item}'] = data_dict['Y_test'][item].values
        comparison_df[f'pred_{item}'] = Y_test_pred[item].values
    
    comparison_df.to_csv(os.path.join(output_dir, 'predictions_comparison.csv'), index=False)
    
    logger.info(f"结果已保存到: {output_dir}")
    
    return results

if __name__ == "__main__":
    results = evaluate_phq9_items() 