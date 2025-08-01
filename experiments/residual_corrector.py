
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from config import RANDOM_SEED, BASE_ESTIMATOR, RESIDUAL_CORRECTOR, CV_FOLDS

class ResidualCorrector:

    def __init__(self, X_train, Y_item_oof_train, y_total_train, item_weights, X_test=None, Y_item_test_pred=None):
        
        self.X_train = X_train
        self.Y_item_oof_train = Y_item_oof_train
        self.y_total_train = y_total_train
        self.item_weights = item_weights
        self.X_test = X_test
        self.Y_item_test_pred = Y_item_test_pred
        self.base_estimator = None
        self.residual_corrector = None

    def train(self):
        
        print("训练残差校正模型...")

        # 训练基估计器
        print("训练基估计器...")
        self.base_estimator = self._train_base_estimator()

        # 生成基预测
        y_total_base_oof = self._predict_base(self.Y_item_oof_train)

        # 计算残差
        residuals = self.y_total_train - y_total_base_oof

        # 训练残差校正模型
        print("训练残差校正模型 (item-level)...")
        self.residual_corrector = self._train_residual_corrector(residuals)

        print("残差校正模型训练完成")
        return self.base_estimator, self.residual_corrector

    def _train_base_estimator(self):
        
        # 构建特征矩阵 (item-level)
        X_base_features = pd.DataFrame(index=self.Y_item_oof_train.index)
        for item_name in self.Y_item_oof_train.columns:
            # 使用加权的问题项得分作为特征
            weight = self.item_weights.get(item_name, 1.0)
            X_base_features[f"weighted_{item_name}"] = self.Y_item_oof_train[item_name] * weight

        # 创建基估计器
        if BASE_ESTIMATOR == 'ridge':
            base_model = Ridge(random_state=RANDOM_SEED)
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
        elif BASE_ESTIMATOR == 'lasso':
            base_model = Lasso(random_state=RANDOM_SEED)
            param_grid = {'alpha': [0.01, 0.1, 1.0]}
        elif BASE_ESTIMATOR == 'elasticnet':
            base_model = ElasticNet(random_state=RANDOM_SEED)
            param_grid = {
                'alpha': [0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8]
            }
        else:
            raise ValueError(f"不支持的基估计器类型: {BASE_ESTIMATOR}")

        # 网格搜索
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=CV_FOLDS,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_base_features, self.y_total_train)

        return grid_search.best_estimator_

    def _train_residual_corrector(self, residuals):
        
        # 构建特征矩阵 (item-level)
        Z_train = pd.DataFrame(index=self.X_train.index)

        # 添加原始声学特征
        for col in self.X_train.columns:
            Z_train[f"acoustic_{col}"] = self.X_train[col]

        # 添加问题项OOF预测作为特征 (item-level)
        for item_name in self.Y_item_oof_train.columns:
            Z_train[f"item_{item_name}"] = self.Y_item_oof_train[item_name]

        important_acoustic_features = self.X_train.columns[:min(5, len(self.X_train.columns))]
        important_items = self.Y_item_oof_train.columns[:min(5, len(self.Y_item_oof_train.columns))]
        
        for acoustic_col in important_acoustic_features:
            for item_name in important_items:
                interaction_name = f"interaction_{acoustic_col}_{item_name}"
                Z_train[interaction_name] = self.X_train[acoustic_col] * self.Y_item_oof_train[item_name]

        # 添加权重特征
        for item_name in self.Y_item_oof_train.columns:
            weight = self.item_weights.get(item_name, 1.0)
            Z_train[f"weight_{item_name}"] = weight

        # 创建残差校正模型
        if RESIDUAL_CORRECTOR == 'ridge':
            corrector_model = Ridge(random_state=RANDOM_SEED)
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
        elif RESIDUAL_CORRECTOR == 'lasso':
            corrector_model = Lasso(random_state=RANDOM_SEED)
            param_grid = {'alpha': [0.01, 0.1, 1.0]}
        elif RESIDUAL_CORRECTOR == 'elasticnet':
            corrector_model = ElasticNet(random_state=RANDOM_SEED)
            param_grid = {
                'alpha': [0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8]
            }
        else:
            raise ValueError(f"不支持的残差校正模型类型: {RESIDUAL_CORRECTOR}")

        # 网格搜索
        grid_search = GridSearchCV(
            corrector_model,
            param_grid,
            cv=CV_FOLDS,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(Z_train, residuals)

        return grid_search.best_estimator_

    def _predict_base(self, Y_item):

        X_base_features = pd.DataFrame(index=Y_item.index)
        for item_name in Y_item.columns:
            weight = self.item_weights.get(item_name, 1.0)
            X_base_features[f"weighted_{item_name}"] = Y_item[item_name] * weight

        # 生成预测
        return self.base_estimator.predict(X_base_features)

    def _predict_residual(self, X, Y_item):

        Z_features = pd.DataFrame(index=X.index)

        # 添加原始声学特征
        for col in X.columns:
            Z_features[f"acoustic_{col}"] = X[col]

        # 添加问题项预测作为特征 (item-level)
        for item_name in Y_item.columns:
            Z_features[f"item_{item_name}"] = Y_item[item_name]

        important_acoustic_features = X.columns[:min(5, len(X.columns))]
        important_items = Y_item.columns[:min(5, len(Y_item.columns))]
        
        for acoustic_col in important_acoustic_features:
            for item_name in important_items:
                interaction_name = f"interaction_{acoustic_col}_{item_name}"
                if acoustic_col in X.columns and item_name in Y_item.columns:
                    Z_features[interaction_name] = X[acoustic_col] * Y_item[item_name]

        # 添加权重特征 (这是缺失的部分)
        for item_name in Y_item.columns:
            Z_features[f"weight_{item_name}"] = self.item_weights.get(item_name, 1.0)

        # 生成预测
        return self.residual_corrector.predict(Z_features)

    def predict(self, X=None, Y_item=None):

        current_X = X if X is not None else self.X_test
        current_Y_item = Y_item if Y_item is not None else self.Y_item_test_pred

        if current_X is None:
            raise ValueError("未指定特征矩阵 (X)，且测试集特征矩阵也为None")
        if current_Y_item is None:
            raise ValueError("未指定问题项预测矩阵 (Y_item)，且测试集问题项预测矩阵也为None")

        print(f"生成总分预测 (item-level, 样本数: {len(current_X)})...")

        # 生成基预测
        y_total_base = self._predict_base(current_Y_item)

        # 生成残差预测
        residual = self._predict_residual(current_X, current_Y_item)

        # 生成最终预测
        y_total_pred = y_total_base + residual

        print("总分预测生成完成 (item-level)")
        return y_total_pred

    def predict_base_only(self, Y_item=None):

        current_Y_item = Y_item if Y_item is not None else self.Y_item_test_pred

        if current_Y_item is None:
            raise ValueError("未指定问题项预测矩阵 (Y_item)，且测试集问题项预测矩阵也为None")

        print(f"生成基估计器预测 (item-level, 样本数: {len(current_Y_item)})...")

        # 仅生成基预测
        y_total_base = self._predict_base(current_Y_item)

        print("基估计器预测生成完成 (item-level)")
        return y_total_base
