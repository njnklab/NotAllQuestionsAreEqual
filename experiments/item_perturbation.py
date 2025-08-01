
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from config import PERTURBATION_SIZE
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ItemPerturbationAnalyzer:


    def __init__(self, X_test, Y_item_test, base_estimator, residual_corrector, item_weights, output_dir, 
                 clinical_data=None, item_descriptions=None):

        self.X_test = X_test
        self.Y_item_test = Y_item_test
        self.base_estimator = base_estimator
        self.residual_corrector = residual_corrector
        self.item_weights = item_weights
        self.output_dir = output_dir
        self.clinical_data = clinical_data
        self.item_descriptions = item_descriptions or {}
        
        # 创建问题项分组信息
        self.item_groups = self._create_item_groups()
        
        # 创建输出目录
        os.makedirs(os.path.join(output_dir, 'ipca'), exist_ok=True)
        
    def _create_item_groups(self):

        groups = {}
        for item in self.Y_item_test.columns:
            prefix = ''.join([c for c in item if not c.isdigit()])
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(item)
        
        return groups

    def analyze(self, perturbation_size=PERTURBATION_SIZE, n_top_items=10, n_trajectories=5):

        print(f"执行问题级扰动贡献分析 (扰动大小: {perturbation_size})...")

        # 获取问题项列表
        items = self.Y_item_test.columns

        # 初始化结果列表
        results = []

        # 对每个样本进行分析
        for i, idx in enumerate(self.X_test.index):
            if i % 100 == 0:
                print(f"分析样本 {i+1}/{len(self.X_test)}...")

            # 获取样本数据
            X_sample = self.X_test.loc[[idx]]
            Y_item_sample = self.Y_item_test.loc[[idx]]

            # 计算原始预测
            y_base_orig = self._predict_base(Y_item_sample)
            y_corr_orig = self._predict_residual(X_sample, Y_item_sample)
            y_total_orig = y_base_orig + y_corr_orig

            # 对每个问题项进行扰动分析
            for item in items:
                # 创建正向扰动的问题项预测
                Y_item_pos = Y_item_sample.copy()
                Y_item_pos[item] += perturbation_size

                # 创建负向扰动的问题项预测
                Y_item_neg = Y_item_sample.copy()
                Y_item_neg[item] -= perturbation_size

                # 计算基础路径调制
                y_base_pos = self._predict_base(Y_item_pos)
                y_base_neg = self._predict_base(Y_item_neg)
                delta_base_pos = y_base_pos - y_base_orig
                delta_base_neg = y_base_neg - y_base_orig

                # 计算校正路径调制
                y_corr_pos = self._predict_residual(X_sample, Y_item_pos)
                y_corr_neg = self._predict_residual(X_sample, Y_item_neg)
                delta_corr_pos = y_corr_pos - y_corr_orig
                delta_corr_neg = y_corr_neg - y_corr_orig

                # 计算总预测调制
                delta_total_pos = delta_base_pos + delta_corr_pos
                delta_total_neg = delta_base_neg + delta_corr_neg

                # 使用中心差分量化敏感度
                S_total = (delta_total_pos - delta_total_neg) / (2 * perturbation_size)
                S_base = (delta_base_pos - delta_base_neg) / (2 * perturbation_size)
                S_corr = (delta_corr_pos - delta_corr_neg) / (2 * perturbation_size)

                # 保存结果
                results.append({
                    'sample_id': idx,
                    'item': item,
                    'S_total': S_total[0],
                    'S_base': S_base[0],
                    'S_corr': S_corr[0]
                })

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 保存结果
        results_df.to_csv(os.path.join(self.output_dir, 'ipca', 'sensitivity.csv'), index=False)

        # 计算平均敏感度
        avg_sensitivity = results_df.groupby('item')[['S_total', 'S_base', 'S_corr']].mean()
        
        # 计算变异系数
        cv_stats = results_df.groupby('item')[['S_total', 'S_base', 'S_corr']].apply(
            lambda x: pd.Series({
                'CV_total': x['S_total'].std() / abs(x['S_total'].mean()) if abs(x['S_total'].mean()) > 1e-5 else np.nan,
                'CV_base': x['S_base'].std() / abs(x['S_base'].mean()) if abs(x['S_base'].mean()) > 1e-5 else np.nan,
                'CV_corr': x['S_corr'].std() / abs(x['S_corr'].mean()) if abs(x['S_corr'].mean()) > 1e-5 else np.nan
            })
        )
        
        # 合并平均敏感度和变异系数
        item_stats = pd.merge(avg_sensitivity, cv_stats, left_index=True, right_index=True)
        item_stats.to_csv(os.path.join(self.output_dir, 'ipca', 'item_stats.csv'))
        
        # 生成4个核心分析展示
        print("生成核心分析展示...")
        
        # 1. 项目重要性综合排名表
        self._create_importance_ranking_table(item_stats, results_df)
        
        # 2. 双路径机制分解图
        self._create_dual_pathway_mechanism_plot(item_stats)
        
        # 3. 样本解释瀑布图组合
        self._create_sample_explanation_waterfall(results_df, item_stats)
        
        # 4. 项目一致性热力矩阵
        self._create_consistency_matrix_heatmap(results_df, item_stats)

        print("问题级扰动贡献分析完成")
        return results_df

    def _predict_base(self, Y_item):

        expected_features = self.base_estimator.feature_names_in_
        X_input_for_base = pd.DataFrame(0.0, index=Y_item.index, columns=expected_features)
        
        is_first_sample_first_item_debug = Y_item.attrs.get('ipca_debug_sample', False)
        debug_item_name_overall = Y_item.attrs.get('ipca_debug_item_name', None)
        
        if is_first_sample_first_item_debug:
            print(f"\n--- DEBUG: _predict_base for sample (overall perturbed item: {debug_item_name_overall}) ---")
            print("Input Y_item for _predict_base (first 5 cols):")
            print(Y_item.iloc[:, :5])

        for item_col_name in Y_item.columns:
            feature_name = f"weighted_{item_col_name}"
            if feature_name in expected_features:
                weight = self.item_weights.get(item_col_name, 1.0)
                X_input_for_base[feature_name] = Y_item[item_col_name] * weight
                
                if is_first_sample_first_item_debug and debug_item_name_overall and item_col_name == debug_item_name_overall:
                    print(f"  DEBUG [_predict_base]: Feature '{feature_name}' for perturbed item '{item_col_name}' = {Y_item[item_col_name].values[0]:.4f} * {weight:.4f} = {X_input_for_base[feature_name].values[0]:.4f}")

        if is_first_sample_first_item_debug:
            X_to_print = X_input_for_base.loc[:, (X_input_for_base != 0).any(axis=0)]
            print("X_input_for_base sent to base_estimator.predict() (showing non-zero cols):")
            print(X_to_print)
            if X_to_print.empty and not X_input_for_base.empty:
                 print("X_input_for_base was all zeros.")
        try:
            prediction = self.base_estimator.predict(X_input_for_base)
            if is_first_sample_first_item_debug:
                print(f"Prediction from base_estimator: {prediction}")
            return prediction
        except Exception as e:
            print(f"基础预测错误 (_predict_base): {e}")
            return np.zeros(len(Y_item))

    def _predict_residual(self, X, Y_item):

        expected_features = self.residual_corrector.feature_names_in_
        Z_input_for_residual = pd.DataFrame(0.0, index=X.index, columns=expected_features)

        is_first_sample_first_item_debug = Y_item.attrs.get('ipca_debug_sample', False)
        debug_item_name_overall = Y_item.attrs.get('ipca_debug_item_name', None)

        if is_first_sample_first_item_debug:
            print(f"\n--- DEBUG: _predict_residual for sample (overall perturbed item: {debug_item_name_overall}) ---")
            print("Input X (acoustic features) for _predict_residual (first 5 cols, if any):")
            print(X.iloc[:, :5])
            print("Input Y_item for _predict_residual (first 5 cols):")
            print(Y_item.iloc[:, :5])

        for col_acoustic in X.columns:
            feature_name = f"acoustic_{col_acoustic}"
            if feature_name in expected_features:
                Z_input_for_residual[feature_name] = X[col_acoustic].values
        for item_col_name in Y_item.columns:
            feature_name = f"item_{item_col_name}"
            if feature_name in expected_features:
                Z_input_for_residual[feature_name] = Y_item[item_col_name]
                
                if is_first_sample_first_item_debug and debug_item_name_overall and item_col_name == debug_item_name_overall:
                    print(f"  DEBUG [_predict_residual]: Feature '{feature_name}' for perturbed item '{item_col_name}' = {Y_item[item_col_name].values[0]:.4f}")

        important_acoustic_features = X.columns[:min(5, len(X.columns))]
        important_items = Y_item.columns[:min(5, len(Y_item.columns))]
        
        for acoustic_col in important_acoustic_features:
            for item_col_name in important_items:
                interaction_name = f"interaction_{acoustic_col}_{item_col_name}"
                if interaction_name in expected_features:
                    if acoustic_col in X.columns and item_col_name in Y_item.columns:
                        Z_input_for_residual[interaction_name] = X[acoustic_col] * Y_item[item_col_name]
                        
                        if is_first_sample_first_item_debug and debug_item_name_overall and item_col_name == debug_item_name_overall:
                            print(f"  DEBUG [_predict_residual]: Interaction feature '{interaction_name}' = {X[acoustic_col].values[0]:.4f} * {Y_item[item_col_name].values[0]:.4f} = {Z_input_for_residual[interaction_name].values[0]:.4f}")

        for item_col_name in Y_item.columns:
            weight_feature_name = f"weight_{item_col_name}"
            if weight_feature_name in expected_features:
                Z_input_for_residual[weight_feature_name] = self.item_weights.get(item_col_name, 1.0)

        if is_first_sample_first_item_debug:
            Z_to_print = Z_input_for_residual.loc[:, (Z_input_for_residual != 0).any(axis=0)]
            print("Z_input_for_residual sent to residual_corrector.predict() (showing non-zero cols):")
            print(Z_to_print)
            if Z_to_print.empty and not Z_input_for_residual.empty:
                print("Z_input_for_residual was all zeros (or only acoustic features were zero if X was all zeros).")
        try:
            prediction = self.residual_corrector.predict(Z_input_for_residual)
            if is_first_sample_first_item_debug:
                print(f"Prediction from residual_corrector: {prediction}")
            return prediction
        except Exception as e:
            print(f"残差预测错误 (_predict_residual): {e}")
            return np.zeros(len(X))

    def _create_importance_ranking_table(self, item_stats, results_df):

        print("生成项目重要性综合排名表...")

        if item_stats.empty:
            print("item_stats为空，跳过项目重要性排名表生成")
            return

        stats_df = item_stats.copy()
        stats_df['abs_S_total'] = stats_df['S_total'].abs()
        
        # 按绝对总敏感度排序
        sorted_items = stats_df.sort_values('abs_S_total', ascending=False)
        
        # 计算路径主导性
        pathway_dominance = []
        for _, row in sorted_items.iterrows():
            s_base = abs(row['S_base'])
            s_corr = abs(row['S_corr'])
            total = s_base + s_corr
            
            if total < 1e-6:
                dominance = "Minimal"
            elif s_base > s_corr * 1.5:
                pct = int(s_base / total * 100)
                dominance = f"Base ({pct}%)"
            elif s_corr > s_base * 1.5:
                pct = int(s_corr / total * 100)
                dominance = f"Correction ({pct}%)"
            else:
                base_pct = int(s_base / total * 100)
                corr_pct = int(s_corr / total * 100)
                dominance = f"Balanced ({base_pct}/{corr_pct})"
            pathway_dominance.append(dominance)
        
        # 计算一致性等级
        consistency_levels = []
        for _, row in sorted_items.iterrows():
            cv_total = row.get('CV_total', 0.5)
            if pd.isna(cv_total) or cv_total == np.inf:
                cv_total = 0.5
            
            if cv_total < 0.3:
                consistency = "High"
            elif cv_total < 0.6:
                consistency = "Medium"
            else:
                consistency = "Low"
            consistency_levels.append(consistency)
        
        # 生成临床标签
        clinical_labels = []
        for i, (idx, row) in enumerate(sorted_items.iterrows()):
            if i < 3:
                label = "Core Depression Indicator" if 'PHQ' in idx else "Primary Anxiety Marker"
            elif i < 8:
                if 'PHQ' in idx:
                    label = "Depressed Mood"
                elif 'GAD' in idx:
                    label = "Anxiety Symptom"
                elif 'ISI' in idx:
                    label = "Sleep Disturbance"
                elif 'PSS' in idx:
                    label = "Stress Response"
                else:
                    label = "Psychological Symptom"
            else:
                label = "Secondary Indicator"
            clinical_labels.append(label)
        
        # 创建排名表
        ranking_table = pd.DataFrame({
            'Rank': range(1, len(sorted_items) + 1),
            'Item': sorted_items.index,
            'Mean_S_total': sorted_items['S_total'].round(3),
            'Mean_S_base': sorted_items['S_base'].round(3),
            'Mean_S_corr': sorted_items['S_corr'].round(3),
            'Pathway_Dominance': pathway_dominance,
            'Consistency': consistency_levels,
            'Clinical_Label': clinical_labels
        })
        
        # 保存文件
        file_path = os.path.join(self.output_dir, 'ipca', 'item_importance_ranking.csv')
        ranking_table.to_csv(file_path, index=False)
        print(f"项目重要性综合排名表已保存至 {file_path}")

    def _create_dual_pathway_mechanism_plot(self, item_stats):

        print("生成双路径机制分解图...")

        if item_stats.empty:
            print("item_stats为空，跳过双路径机制分解图生成")
            return
        
        stats_df = item_stats.copy()
        
        # 使用Set2调色板创建项目分组颜色映射
        set2_colors = plt.cm.Set2(np.linspace(0, 1, 8))
        color_map = {}
        for i, item in enumerate(stats_df.index):
            if 'PHQ' in item:
                color_map[item] = set2_colors[0]
            elif 'GAD' in item:
                color_map[item] = set2_colors[1]
            elif 'ISI' in item:
                color_map[item] = set2_colors[2]
            elif 'PSS' in item:
                color_map[item] = set2_colors[3]
            else:
                color_map[item] = set2_colors[4]
        
        # 使用兼容的样式
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # 绘制散点图
        for item in stats_df.index:
            s_base = stats_df.loc[item, 'S_base']
            s_corr = stats_df.loc[item, 'S_corr'] 
            s_total = abs(stats_df.loc[item, 'S_total'])
            
            ax.scatter(s_base, s_corr, 
                      s=s_total*500+50,
                      color=color_map[item], 
                      alpha=0.7, 
                      edgecolor='black', 
                      linewidth=0.5,
                      label=item)
            
            # 在每个点附近标注题目名称
            ax.annotate(item, (s_base, s_corr), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=15, ha='left', va='bottom', fontweight='bold', fontfamily='Arial',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 添加象限线
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        
        # Q1: 右上角
        ax.text(x_lim[1]*0.95, y_lim[1]*0.95, 'Q1: Synergistic Effects\n(High Base + High Correction)', 
                ha='right', va='top', fontsize=10, fontweight='bold', fontfamily='Arial',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Q2: 左上角
        ax.text(x_lim[0]*0.95, y_lim[1]*0.95, 'Q2: Correction-Driven\n(Low Base + High Correction)', 
                ha='left', va='top', fontsize=10, fontweight='bold', fontfamily='Arial',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        # Q3: 左下角
        ax.text(x_lim[0]*0.95, y_lim[0]*0.95, 'Q3: Minimal Effects\n(Low Base + Low Correction)', 
                ha='left', va='bottom', fontsize=10, fontweight='bold', fontfamily='Arial',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        # Q4: 右下角
        ax.text(x_lim[1]*0.95, y_lim[0]*0.95, 'Q4: Base-Driven\n(High Base + Low Correction)', 
                ha='right', va='bottom', fontsize=10, fontweight='bold', fontfamily='Arial',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsalmon', alpha=0.7))
        
        ax.set_xlabel('Base Pathway Sensitivity', fontsize=16, fontweight='bold', fontfamily='Arial')
        ax.set_ylabel('Correction Pathway Sensitivity', fontsize=16, fontweight='bold', fontfamily='Arial')

        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        file_path = os.path.join(self.output_dir, 'ipca', 'dual_pathway_mechanism.jpg')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"双路径机制分解图已保存至 {file_path}")
        plt.close()

    def _create_sample_explanation_waterfall(self, results_df, item_stats):

        if results_df.empty:
            print("results_df为空，跳过样本解释瀑布图生成")
            return
        
        # 选择3个代表性样本
        sample_ids = results_df['sample_id'].unique()
        if len(sample_ids) < 3:
            print(f"样本数量不足({len(sample_ids)})，跳过瀑布图生成")
            return
        
        # 改进样本选择逻辑：随机选择3个不同的样本来确保多样性
        np.random.seed(42)
        selected_samples = np.random.choice(sample_ids, size=min(3, len(sample_ids)), replace=False).tolist()
        
        # 根据总敏感度对选中的样本进行排序，以便标记为轻度、中度、重度
        sample_totals = results_df[results_df['sample_id'].isin(selected_samples)].groupby('sample_id')['S_total'].sum().abs()
        selected_samples = sample_totals.sort_values().index.tolist()
        
        sample_labels = ['Light Severity', 'Medium Severity', 'High Severity']
        
        # 为每个样本创建单独的竖图
        for i, (sample_id, label) in enumerate(zip(selected_samples, sample_labels)):
            # 获取该样本的数据
            sample_data = results_df[results_df['sample_id'] == sample_id].copy()
            if sample_data.empty:
                continue
                
            # 按敏感度排序，取Top 10
            sample_data = sample_data.sort_values('S_total', key=abs, ascending=False).head(10)
            
            items = sample_data['item'].values
            s_totals = sample_data['S_total'].values
            s_bases = sample_data['S_base'].values
            s_corrs = sample_data['S_corr'].values
            
            # 计算累积值
            cumulative = np.cumsum(s_totals)
            
            # 创建竖图
            fig, ax = plt.subplots(figsize=(2.5, 5))
            
            # 绘制瀑布图
            x_pos = np.arange(len(items))
            
            # 基础贡献
            ax.bar(x_pos, s_bases, color='#4c72b0', alpha=0.8, label='Base Pathway')
            # 校正贡献  
            ax.bar(x_pos, s_corrs, bottom=s_bases, color='#dd8453', alpha=0.8, label='Correction Pathway')
            
            # 添加累积线
            ax.plot(x_pos, cumulative, color='#c44e52', marker='o', linewidth=2, markersize=6, label='Cumulative Total')
            
            # 设置标签
            ax.set_title(f'{label} {sample_id}', fontsize=11, fontweight='bold', fontfamily='Arial')
            ax.set_xlabel('Top Contributing Items', fontsize=11, fontweight='bold', fontfamily='Arial')
            ax.set_ylabel('Sensitivity Contribution', fontsize=11, fontweight='bold', fontfamily='Arial')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(items, rotation=90, ha='right', fontsize=11, fontfamily='Arial')
            
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.5)
            
            plt.tight_layout()
            
            # 保存单独的图像
            file_path = os.path.join(self.output_dir, 'ipca', f'sample_explanation_waterfall_{label.lower().replace(" ", "_")}.jpg')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"{label}样本解释瀑布图已保存至 {file_path}")
            plt.close()

    def _create_consistency_matrix_heatmap(self, results_df, item_stats):
        """
        创建项目一致性热力矩阵
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            完整的敏感度分析结果
        item_stats : pandas.DataFrame
            项目统计数据
        """
        print("生成项目一致性热力矩阵...")

        if results_df.empty:
            print("results_df为空，跳过项目一致性热力矩阵生成")
            return
        
        # 数据透视：样本为行，项目为列
        pivot_data = results_df.pivot(index='sample_id', columns='item', values='S_total')
        pivot_data = pivot_data.fillna(0)
        
        if pivot_data.shape[1] < 2:
            print("项目数量不足，跳过一致性矩阵生成")
            return
        
        # 按重要性排序项目
        item_order = item_stats.reindex(pivot_data.columns)['S_total'].abs().sort_values(ascending=False).index
        pivot_data = pivot_data[item_order]
        
        # 将样本按总分分组
        sample_totals = pivot_data.sum(axis=1)
        n_samples = len(sample_totals)
        
        # 分为三组：轻度、中度、重度
        terciles = np.percentile(sample_totals, [33, 67])
        
        low_samples = sample_totals[sample_totals <= terciles[0]].index
        med_samples = sample_totals[(sample_totals > terciles[0]) & (sample_totals <= terciles[1])].index
        high_samples = sample_totals[sample_totals > terciles[1]].index
        
        # 计算各组平均敏感度
        low_avg = pivot_data.loc[low_samples].mean()
        med_avg = pivot_data.loc[med_samples].mean()
        high_avg = pivot_data.loc[high_samples].mean()
        
        # 创建矩阵
        severity_matrix = pd.DataFrame({
            'Low Severity': low_avg,
            'Medium Severity': med_avg, 
            'High Severity': high_avg
        })
        
        # 添加一致性评分列
        consistency_scores = []
        for item in severity_matrix.index:
            values = severity_matrix.loc[item].values
            cv = np.std(values) / (np.mean(values) + 1e-6)  # 避免除零
            
            if cv < 0.3:
                score = "HIGH"
            elif cv < 0.6:
                score = "MEDIUM"
            else:
                score = "LOW"
            consistency_scores.append(score)
        
        severity_matrix['Consistency'] = consistency_scores
        
        # 绘制热力图（去掉右边的Consistency Scores）
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(1, 1, figsize=(5, 10))
        
        # 主热力图
        numeric_data = severity_matrix.iloc[:, :3]  # 只包含数值列
        sns.heatmap(numeric_data, annot=True, fmt='.3f', cmap='Reds', ax=ax)
        ax.set_title('Item Sensitivity Across Severity Levels', fontsize=15, fontweight='bold', fontfamily='Arial')
        ax.set_xlabel('Severity Groups', fontsize=14, fontweight='bold', fontfamily='Arial')
        ax.set_ylabel('Items (Ranked by Importance)', fontsize=14, fontweight='bold', fontfamily='Arial')
        
        plt.tight_layout()
        
        # 保存图像
        file_path = os.path.join(self.output_dir, 'ipca', 'consistency_matrix_heatmap.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"项目一致性热力矩阵已保存至 {file_path}")
        plt.close()

    def _create_impact_panorama(self, item_stats, n_top_items=15):
        """
        为了兼容性保留的旧方法，实际调用新的_create_importance_ranking_table方法
        """
        print("警告: 使用了旧的_create_impact_panorama方法，建议更新代码使用_create_importance_ranking_table")
        return self._create_importance_ranking_table(item_stats, None)
        
    def _create_pathway_analysis(self, item_stats):
        """
        为了兼容性保留的旧方法，实际调用新的_create_dual_pathway_mechanism_plot方法
        """
        print("警告: 使用了旧的_create_pathway_analysis方法，建议更新代码使用_create_dual_pathway_mechanism_plot")
        return self._create_dual_pathway_mechanism_plot(item_stats)
        
    def _create_sensitivity_network(self, results_df, item_stats):
        """
        为了兼容性保留的旧方法，实际调用新的_create_sample_explanation_waterfall方法
        """
        print("警告: 使用了旧的_create_sensitivity_network方法，建议更新代码使用_create_sample_explanation_waterfall")
        return self._create_sample_explanation_waterfall(results_df, item_stats)
        
    def _create_clinical_value_assessment(self, item_stats):
        """
        为了兼容性保留的旧方法，实际调用新的_create_consistency_matrix_heatmap方法
        """
        print("警告: 使用了旧的_create_clinical_value_assessment方法，建议更新代码使用_create_consistency_matrix_heatmap")
        return self._create_consistency_matrix_heatmap(None, item_stats)

    def generate_sample_rank_csv(self, results_df):
        """
        生成所有样本的题目rank顺序CSV文件
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            完整的敏感度分析结果，包含所有样本的item敏感度数据
        """
        print("生成所有样本的题目rank顺序CSV...")
        
        if results_df.empty:
            print("results_df为空，跳过rank CSV生成")
            return
        
        # 获取所有唯一的样本ID和item
        sample_ids = sorted(results_df['sample_id'].unique())
        all_items = sorted(results_df['item'].unique())
        
        # 创建结果列表
        rank_data = []
        
        for sample_id in sample_ids:
            # 获取该样本的数据
            sample_data = results_df[results_df['sample_id'] == sample_id].copy()
            
            if sample_data.empty:
                continue
            
            # 按S_total的绝对值降序排序
            sample_data = sample_data.sort_values('S_total', key=abs, ascending=False)
            
            # 创建该样本的rank字典
            sample_rank = {'id': sample_id}
            
            # 为每个item添加rank
            for i, (_, row) in enumerate(sample_data.iterrows(), 1):
                rank_key = f"rank{i}"
                sample_rank[rank_key] = row['item']
            
            # 如果某些item没有数据，用None填充剩余的rank
            max_ranks = len(all_items)
            for i in range(len(sample_data) + 1, max_ranks + 1):
                rank_key = f"rank{i}"
                sample_rank[rank_key] = None
                
            rank_data.append(sample_rank)
        
        # 转换为DataFrame
        rank_df = pd.DataFrame(rank_data)
        
        # 保存CSV文件
        csv_path = os.path.join(self.output_dir, 'ipca', 'sample_item_ranks.csv')
        rank_df.to_csv(csv_path, index=False)
        
        print(f"样本题目rank顺序CSV已保存至 {csv_path}")
        print(f"共处理 {len(sample_ids)} 个样本，每个样本最多 {len(all_items)} 个题目")
        
        return rank_df

    def create_single_sample_explanation(self, sample_id, X_sample, Y_item_sample, 
                                       save_path=None, top_n=10, 
                                       perturbation_size=None):
        """
        为单个样本创建explanation瀑布图
        
        Parameters
        ----------
        sample_id : int or str
            样本ID
        X_sample : pandas.DataFrame
            单个样本的特征数据 (1行)
        Y_item_sample : pandas.DataFrame  
            单个样本的item预测数据 (1行)
        save_path : str, optional
            图片保存路径，如果为None则保存到默认位置
        top_n : int, default=10
            显示top N个最重要的item
        perturbation_size : float, optional
            扰动大小，如果为None则使用默认值
            
        Returns
        -------
        str
            保存的图片路径
        """
        print(f"为样本 {sample_id} 创建单个explanation图...")
        
        if perturbation_size is None:
            perturbation_size = PERTURBATION_SIZE
            
        # 计算原始预测
        y_base_orig = self._predict_base(Y_item_sample)
        y_corr_orig = self._predict_residual(X_sample, Y_item_sample)
        y_total_orig = y_base_orig + y_corr_orig
        
        # 对每个问题项进行扰动分析
        item_results = []
        for item in Y_item_sample.columns:
            # 创建正向扰动的问题项预测
            Y_item_pos = Y_item_sample.copy()
            Y_item_pos[item] += perturbation_size
            
            # 创建负向扰动的问题项预测
            Y_item_neg = Y_item_sample.copy()
            Y_item_neg[item] -= perturbation_size
            
            # 计算基础路径调制
            y_base_pos = self._predict_base(Y_item_pos)
            y_base_neg = self._predict_base(Y_item_neg)
            delta_base_pos = y_base_pos - y_base_orig
            delta_base_neg = y_base_neg - y_base_orig
            
            # 计算校正路径调制
            y_corr_pos = self._predict_residual(X_sample, Y_item_pos)
            y_corr_neg = self._predict_residual(X_sample, Y_item_neg)
            delta_corr_pos = y_corr_pos - y_corr_orig
            delta_corr_neg = y_corr_neg - y_corr_orig
            
            # 计算总预测调制
            delta_total_pos = delta_base_pos + delta_corr_pos
            delta_total_neg = delta_base_neg + delta_corr_neg
            
            # 使用中心差分量化敏感度
            S_total = (delta_total_pos - delta_total_neg) / (2 * perturbation_size)
            S_base = (delta_base_pos - delta_base_neg) / (2 * perturbation_size)
            S_corr = (delta_corr_pos - delta_corr_neg) / (2 * perturbation_size)
            
            # 确保是标量值
            S_total = S_total[0] if isinstance(S_total, np.ndarray) and S_total.size == 1 else S_total
            S_base = S_base[0] if isinstance(S_base, np.ndarray) and S_base.size == 1 else S_base
            S_corr = S_corr[0] if isinstance(S_corr, np.ndarray) and S_corr.size == 1 else S_corr
            
            item_results.append({
                'item': item,
                'S_total': S_total,
                'S_base': S_base,
                'S_corr': S_corr
            })
        
        # 转换为DataFrame并排序
        sample_data = pd.DataFrame(item_results)
        sample_data = sample_data.sort_values('S_total', key=abs, ascending=False).head(top_n)
        
        # 获取数据
        items = sample_data['item'].values
        s_totals = sample_data['S_total'].values
        s_bases = sample_data['S_base'].values
        s_corrs = sample_data['S_corr'].values
        
        # 计算累积值
        cumulative = np.cumsum(s_totals)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(4, 7))
        
        # 绘制瀑布图
        x_pos = np.arange(len(items))
        
        # 基础贡献
        ax.bar(x_pos, s_bases, color='cornflowerblue', alpha=0.8, label='Base Pathway')
        # 校正贡献  
        ax.bar(x_pos, s_corrs, bottom=s_bases, color='lightsalmon', alpha=0.8, label='Correction Pathway')
        
        # 添加累积线
        ax.plot(x_pos, cumulative, color='red', marker='o', linewidth=2, markersize=6, label='Cumulative Total')
        
        # 设置标签
        ax.set_title(f'Sample Explanation (ID: {sample_id})', fontsize=14, fontweight='bold', fontfamily='Arial')
        ax.set_xlabel('Top Contributing Items', fontsize=13, fontweight='bold', fontfamily='Arial')
        ax.set_ylabel('Sensitivity Contribution', fontsize=13, fontweight='bold', fontfamily='Arial')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(items, rotation=45, ha='right', fontsize=12, fontfamily='Arial')
        
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        # 确定保存路径
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'ipca', f'sample_explanation_{sample_id}.jpg')
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"样本 {sample_id} 的explanation图已保存至 {save_path}")
        plt.close()
        
        return save_path
