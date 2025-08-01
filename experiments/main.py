#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LIRA model main program
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import pickle
import sys
import os
import multiprocessing as mp
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from LIRA.data_processor import DataProcessor
from LIRA.weight_calculator import WeightCalculator
from LIRA.item_predictor import ItemPredictor
from LIRA.residual_corrector import ResidualCorrector
from LIRA.evaluator import Evaluator
from LIRA.item_perturbation import ItemPerturbationAnalyzer
from config import CALCULATE_MODULE_CONTROLLABILITY, PERTURBATION_SIZE

ACOUSTIC_FEATURES_PATH = '/home/a001/xuxiao/LIRA/dataset/CS-NRAC-E.csv'
QUESTIONNAIRE_PATH = '/home/a001/xuxiao/LIRA/dataset/raw_info.csv'

def save_model(model, path):
    """Save model to file"""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def analyze_sample_batch(batch_indices, X_test_data, Y_test_pred_data, ipca_analyzer, current_perturbation_size):
    """Analyze a batch of samples"""
    batch_results_list = []
    
    for i_loop, idx_loop in enumerate(batch_indices):
        is_first_sample_for_debug = (idx_loop == X_test_data.index[0]) if X_test_data.index.size > 0 else False

        if i_loop % 10 == 0:
            process_id = os.getpid()
            print(f"Process {process_id}: Analyzing sample {i_loop+1}/{len(batch_indices)} (index: {idx_loop})...")
            
        X_sample = X_test_data.loc[[idx_loop]]
        Y_item_sample = Y_test_pred_data.loc[[idx_loop]]
        
        y_base_orig = ipca_analyzer._predict_base(Y_item_sample)
        y_corr_orig = ipca_analyzer._predict_residual(X_sample, Y_item_sample)
        y_total_orig = y_base_orig + y_corr_orig
        
        for item_col in Y_test_pred_data.columns:
            Y_item_pos = Y_item_sample.copy()
            Y_item_pos[item_col] += current_perturbation_size
            
            Y_item_neg = Y_item_sample.copy()
            Y_item_neg[item_col] -= current_perturbation_size
            
            y_base_pos = ipca_analyzer._predict_base(Y_item_pos)
            y_base_neg = ipca_analyzer._predict_base(Y_item_neg)
            delta_base_pos = y_base_pos - y_base_orig
            delta_base_neg = y_base_neg - y_base_orig
            
            y_corr_pos = ipca_analyzer._predict_residual(X_sample, Y_item_pos)
            y_corr_neg = ipca_analyzer._predict_residual(X_sample, Y_item_neg)
            delta_corr_pos = y_corr_pos - y_corr_orig
            delta_corr_neg = y_corr_neg - y_corr_orig
            
            delta_total_pos = delta_base_pos + delta_corr_pos
            delta_total_neg = delta_base_neg + delta_corr_neg
            
            S_total = (delta_total_pos - delta_total_neg) / (2 * current_perturbation_size)
            S_base = (delta_base_pos - delta_base_neg) / (2 * current_perturbation_size)
            S_corr = (delta_corr_pos - delta_corr_neg) / (2 * current_perturbation_size)

            batch_results_list.append({
                'sample_id': idx_loop,
                'item': item_col,
                'S_total': S_total,
                'S_base': S_base,
                'S_corr': S_corr,
                'y_total_orig': y_total_orig,
                'y_base_orig': y_base_orig,
                'y_corr_orig': y_corr_orig
            })

    return batch_results_list

def main():
    """Main function"""
    acoustic_path = '/home/a001/xuxiao/LIRA/dataset/CS-NRAC-E.csv'
    questionnaire_path = '/home/a001/xuxiao/LIRA/dataset/raw_info.csv'
    output_dir = '/home/a001/xuxiao/LIRA/results/LIRA'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ipca'), exist_ok=True)

    start_time = time.time()

    print("\n=== Data Processing ===")
    data_processor = DataProcessor(acoustic_path, questionnaire_path)
    data_dict = data_processor.process()

    print("\n=== Network Analysis and Weight Calculation ===")
    weight_calculator = WeightCalculator(data_dict['Y_train'])
    item_weights, subscale_weights = weight_calculator.calculate_weights(calculate_module_controllability=CALCULATE_MODULE_CONTROLLABILITY)

    item_weights.to_csv(os.path.join(output_dir, 'item_weights.csv'))
    print(f"Item weights (CF values) saved to {os.path.join(output_dir, 'item_weights.csv')}")

    print("\n=== Item-level Prediction and Subscale Aggregation ===")
    item_predictor = ItemPredictor(
        data_dict['X_train'],
        data_dict['Y_train'],
        data_dict['X_test'],
        item_weights
    )

    item_predictor.train_models()

    Y_oof = item_predictor.generate_oof_predictions()

    Y_test_pred = item_predictor.predict()

    print("\n=== Structured Residual Correction (item-level) ===")
    residual_corrector_instance = ResidualCorrector(
        data_dict['X_train'],
        Y_oof,
        data_dict['y_total_train'],
        item_weights,
        data_dict['X_test'],
        Y_test_pred
    )

    base_estimator, residual_corrector_model = residual_corrector_instance.train()

    y_total_test_pred = residual_corrector_instance.predict()

    print("\n=== Evaluation ===")
    evaluator = Evaluator(
        data_dict['Y_test'],
        data_dict['y_total_test'],
        y_total_test_pred
    )

    metrics = evaluator.evaluate()
    
    print("\n=== Item-level perturbation contribution analysis (multi-process) ===")
    
    ipca_analyzer_instance = ItemPerturbationAnalyzer(
        data_dict['X_test'],
        Y_test_pred,
        base_estimator,
        residual_corrector_model,
        item_weights,
        output_dir
    )
    
    num_cores = mp.cpu_count()
    print(f"Detected {num_cores} CPU cores.")
    num_workers = max(1, num_cores - 1 if num_cores > 1 else 1) 
    print(f"Will use {num_workers} threads for parallel IPCA calculation.")
    
    sample_indices = data_dict['X_test'].index.tolist()
    if not sample_indices:
        print("Error: No samples available for IPCA analysis. Please check X_test data.")
        return

    batch_size = max(1, len(sample_indices) // num_workers if len(sample_indices) >= num_workers else 1)
    batches = [sample_indices[i:i + batch_size] for i in range(0, len(sample_indices), batch_size)]
    
    print(f"Total IPCA samples: {len(sample_indices)}")
    print(f"IPCA samples per batch: {batch_size}")
    print(f"Total IPCA batches: {len(batches)}")
    
    analyze_func_partial = partial(
        analyze_sample_batch, 
        X_test_data=data_dict['X_test'],
        Y_test_pred_data=Y_test_pred,
        ipca_analyzer=ipca_analyzer_instance, 
        current_perturbation_size=PERTURBATION_SIZE
    )
    
    print("\nStarting parallel IPCA sample analysis...")
    all_ipca_results_list = []
    if batches:
        with mp.Pool(processes=num_workers) as pool:
            list_of_lists_results = pool.map(analyze_func_partial, batches)
        
        for single_batch_results in list_of_lists_results:
            all_ipca_results_list.extend(single_batch_results)
        print("Parallel IPCA analysis completed.")
    else:
        print("No batch data available for IPCA analysis.")

    if not all_ipca_results_list:
        print("Error: IPCA analysis did not produce any results. Please check input data and perturbation logic.")
    else:
        results_df = pd.DataFrame(all_ipca_results_list)
        
        print("\nSaving detailed IPCA sensitivity analysis results...")
        results_df.to_csv(os.path.join(output_dir, 'ipca', 'sensitivity.csv'), index=False)
        
        print("Calculating IPCA statistics (average sensitivity, CV)...")
        avg_sensitivity = results_df.groupby('item')[['S_total', 'S_base', 'S_corr']].mean()
        
        cv_stats = results_df.groupby('item')[['S_total', 'S_base', 'S_corr']].apply(
            lambda x: pd.Series({
                'CV_total': x['S_total'].std() / abs(x['S_total'].mean()) if abs(x['S_total'].mean()) > 1e-6 else np.nan,
                'CV_base': x['S_base'].std() / abs(x['S_base'].mean()) if abs(x['S_base'].mean()) > 1e-6 else np.nan,
                'CV_corr': x['S_corr'].std() / abs(x['S_corr'].mean()) if abs(x['S_corr'].mean()) > 1e-6 else np.nan
            })
        ).reset_index()
        
        item_stats = pd.merge(avg_sensitivity.reset_index(), cv_stats, on='item').set_index('item')

        item_stats.to_csv(os.path.join(output_dir, 'ipca', 'item_stats.csv'))
        
        print("\nGenerating IPCA analysis charts...")
        if not item_stats.empty:
            ipca_analyzer_instance._create_importance_ranking_table(item_stats, results_df)
            ipca_analyzer_instance._create_dual_pathway_mechanism_plot(item_stats)
            ipca_analyzer_instance._create_consistency_matrix_heatmap(results_df, item_stats)
        else:
            print("item_stats is empty, skipping chart generation.")

        if not results_df.empty:
            ipca_analyzer_instance._create_sample_explanation_waterfall(results_df, item_stats)
        else:
            print("results_df is empty, skipping sample explanation waterfall plot generation.")

        if not results_df.empty:
            print("\n=== Generating sample item rank order CSV ===")
            ipca_analyzer_instance.generate_sample_rank_csv(results_df)
        else:
            print("results_df is empty, skipping sample rank CSV generation.")

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    original_perturbation_size = PERTURBATION_SIZE
    PERTURBATION_SIZE = original_perturbation_size * 10 
    print(f"Note: PERTURBATION_SIZE has been temporarily increased to {PERTURBATION_SIZE} for testing.")
    
    main()
