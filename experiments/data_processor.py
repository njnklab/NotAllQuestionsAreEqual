#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing module
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import RANDOM_SEED, SUBSCALES, TEST_SIZE, VAL_SIZE, OUTLIER_THRESHOLD, APPLY_PCA, PCA_VARIANCE
from sdi_filter import SDIFilter

class DataProcessor:
    """Data processing class"""

    def __init__(self, acoustic_path, questionnaire_path):
        """Initialize data processor"""
        self.acoustic_path = acoustic_path
        self.questionnaire_path = questionnaire_path

    def load_data(self):
        """Load data"""
        print("Loading acoustic feature data...")
        X = pd.read_csv(self.acoustic_path)

        print("Loading questionnaire data...")
        Y = pd.read_csv(self.questionnaire_path)

        print(f"Original acoustic feature data size: {X.shape}")
        print(f"Original questionnaire data size: {Y.shape}")

        common_ids = set(X['id']).intersection(set(Y['id']))
        print(f"Acoustic feature data and questionnaire data have {len(common_ids)} common IDs")

        X = X[X['id'].isin(common_ids)].set_index('id')
        Y = Y[Y['id'].isin(common_ids)].set_index('id')

        common_indices = X.index.intersection(Y.index)
        X = X.loc[common_indices]
        Y = Y.loc[common_indices]

        print(f"Filtered acoustic feature data size: {X.shape}")
        print(f"Filtered questionnaire data size: {Y.shape}")

        print(f"Number of NaN values in acoustic feature data: {X.isna().sum().sum()}")
        print(f"Number of NaN values in questionnaire data: {Y.isna().sum().sum()}")

        X = X.dropna()
        Y = Y.loc[X.index]

        if Y.index.duplicated().any():
            print(f"Warning: Duplicate values in Y.index, removing duplicate rows...")
            Y = Y[~Y.index.duplicated(keep='first')]

        common_indices = X.index.intersection(Y.index)
        X = X.loc[common_indices]
        Y = Y.loc[common_indices]

        print(f"Processed acoustic feature data size: {X.shape}")
        print(f"Processed questionnaire data size: {Y.shape}")

        print("Calculating subscale scores...")
        Y_sub = pd.DataFrame(index=Y.index)

        for subscale_name, items in SUBSCALES.items():
            valid_items = [item for item in items if item in Y.columns]
            if len(valid_items) != len(items):
                missing_items = set(items) - set(valid_items)
                print(f"Warning: Subscale {subscale_name} missing items: {missing_items}")

            Y_sub[subscale_name] = Y[valid_items].sum(axis=1)

        print("Calculating total score...")
        y_total = Y_sub.sum(axis=1)

        return X, Y, Y_sub, y_total

    def split_data(self, X, Y, Y_sub, y_total):
        """Split data into train, validation, and test sets"""
        print(f"Splitting dataset (test size: {TEST_SIZE}, validation size: {VAL_SIZE})...")

        X_temp, X_test, Y_temp, Y_test, Y_sub_temp, Y_sub_test, y_total_temp, y_total_test = train_test_split(
            X, Y, Y_sub, y_total, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, Y_train, Y_val, Y_sub_train, Y_sub_val, y_total_train, y_total_val = train_test_split(
            X_temp, Y_temp, Y_sub_temp, y_total_temp, test_size=val_size_adjusted, random_state=RANDOM_SEED
        )

        print(f"Dataset split complete - Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")

        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test,
            'Y_sub_train': Y_sub_train, 'Y_sub_val': Y_sub_val, 'Y_sub_test': Y_sub_test,
            'y_total_train': y_total_train, 'y_total_val': y_total_val, 'y_total_test': y_total_test
        }

    def process(self):
        """Main data processing pipeline"""
        X, Y, Y_sub, y_total = self.load_data()

        if hasattr(self, 'apply_sdi_filter') and self.apply_sdi_filter:
            print("Applying SDI filter...")
            sdi_filter = SDIFilter()
            X = sdi_filter.filter(X)

        if APPLY_PCA:
            print(f"Applying PCA with {PCA_VARIANCE} variance...")
            pca = PCA(n_components=PCA_VARIANCE)
            X = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
            print(f"PCA applied. New feature count: {X.shape[1]}")

        data_dict = self.split_data(X, Y, Y_sub, y_total)

        return data_dict
