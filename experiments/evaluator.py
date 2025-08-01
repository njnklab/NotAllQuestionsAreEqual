
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from scipy.stats import pearsonr
from config import SUBSCALES

def non_negative_r2_score(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    return max(0.0, r2)

class Evaluator:

    def __init__(self, Y_test, Y_test_pred, y_total_test, y_total_test_pred, output_dir):

        self.Y_test = Y_test
        self.Y_test_pred = Y_test_pred
        self.y_total_test = y_total_test
        self.y_total_test_pred = y_total_test_pred
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self):
        print("执行评估...")

        metrics = {
            'item_level': {},
            'total_level': {}
        }

        print("评估项目级预测...")
        self.valid_items_for_eval = [item for subscale_items in SUBSCALES.values() for item in subscale_items
                                     if item in self.Y_test.columns and item in self.Y_test_pred.columns]

        for item in self.valid_items_for_eval:
            y_true = self.Y_test[item]
            y_pred = self.Y_test_pred[item]

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            non_neg_r2 = non_negative_r2_score(y_true, y_pred)
            exp_var = explained_variance_score(y_true, y_pred)
            pearson_r, p_value = pearsonr(y_true, y_pred)

            # 保存评估指标
            metrics['item_level'][item] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'non_neg_r2': non_neg_r2,
                'exp_var': exp_var,
                'pearson_r': pearson_r,
                'p_value': p_value
            }

        # 计算项目级平均指标
        metrics['item_level']['average'] = {
            'rmse': np.mean([m['rmse'] for m in metrics['item_level'].values() if isinstance(m, dict)]),
            'mae': np.mean([m['mae'] for m in metrics['item_level'].values() if isinstance(m, dict)]),
            'r2': np.mean([m['r2'] for m in metrics['item_level'].values() if isinstance(m, dict)]),
            'non_neg_r2': np.mean([m['non_neg_r2'] for m in metrics['item_level'].values() if isinstance(m, dict)]),
            'exp_var': np.mean([m['exp_var'] for m in metrics['item_level'].values() if isinstance(m, dict)]),
            'pearson_r': np.mean([m['pearson_r'] for m in metrics['item_level'].values() if isinstance(m, dict)])
        }

        print("评估总分级预测...")
        y_true = self.y_total_test
        y_pred = self.y_total_test_pred

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        non_neg_r2 = non_negative_r2_score(y_true, y_pred)
        exp_var = explained_variance_score(y_true, y_pred)
        pearson_r, p_value = pearsonr(y_true, y_pred)

        metrics['total_level'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'non_neg_r2': non_neg_r2,
            'exp_var': exp_var,
            'pearson_r': pearson_r,
            'p_value': p_value
        }
        
        self._save_metrics(metrics)

        self._visualize_results()

        self._generate_report(metrics)

        print("评估完成")
        return metrics
    def _save_metrics(self, metrics):
        item_metrics = pd.DataFrame([
            {
                'item': item,
                'rmse': m['rmse'],
                'mae': m['mae'],
                'r2': m['r2'],
                'non_neg_r2': m['non_neg_r2'],
                'exp_var': m['exp_var'],
                'pearson_r': m['pearson_r'],
                'p_value': m['p_value'] if 'p_value' in m else np.nan
            }
            for item, m in metrics['item_level'].items() if isinstance(m, dict)
        ])
        item_metrics.to_csv(os.path.join(self.output_dir, 'item_metrics.csv'), index=False)

        # 保存总分级指标
        total_metrics = pd.DataFrame([
            {
                'metric': metric,
                'value': value
            }
            for metric, value in metrics['total_level'].items()
        ])
        total_metrics.to_csv(os.path.join(self.output_dir, 'total_metrics.csv'), index=False)

    def _visualize_results(self):
        sns.set(style="whitegrid")

        # 可视化总分预测
        joint_plot_fig = sns.jointplot(x=self.y_total_test, y=self.y_total_test_pred, kind="reg", height=8)
        joint_plot_fig.ax_joint.set_xlabel('True Total Score', fontsize=18, fontweight='bold', fontfamily='Arial')
        joint_plot_fig.ax_joint.set_ylabel('Predicted Total Score', fontsize=18, fontweight='bold', fontfamily='Arial')
        
        # 增大坐标轴刻度标签的字体大小
        joint_plot_fig.ax_joint.tick_params(axis='both', which='major', labelsize=17)
        
        joint_plot_fig.savefig(os.path.join(self.output_dir, 'total_scatter.jpg'), dpi=300, bbox_inches='tight')
        plt.close(joint_plot_fig.fig)

        # 可视化项目级Pearson相关系数
        print("可视化项目级Pearson相关系数...")
        item_pearson = {}
        for item in self.valid_items_for_eval:
            # 计算 Pearson 相关系数
            pearson_r, _ = pearsonr(self.Y_test[item], self.Y_test_pred[item])
            item_pearson[item] = pearson_r

        # 创建数据框用于绘图
        plot_data = pd.Series(item_pearson).reset_index()
        plot_data.columns = ['item', 'pearson_r']

        # 提取子量表信息并添加到数据框
        plot_data['subscale'] = plot_data['item'].apply(lambda x: next((s for s, items in SUBSCALES.items() if x in items), 'Unknown'))

        # 按子量表和项目排序数据，以便绘图时保持一致顺序
        plot_data = plot_data.sort_values(by=['subscale', 'item']).reset_index(drop=True)

        # 创建图形和坐标轴
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # 绘制条形图
        bars = sns.barplot(x='item', y='pearson_r', hue='subscale', data=plot_data, ax=ax)
        
        # 设置坐标轴标签，保持与总分图一致的字体设置
        ax.set_xlabel('Item', fontsize=18, fontweight='bold', fontfamily='Arial')
        ax.set_ylabel('Pearson Correlation Coefficient', fontsize=18, fontweight='bold', fontfamily='Arial')
        
        # 增大坐标轴刻度标签的字体大小，与总分图一致
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        # 调整图例
        plt.legend(fontsize=14, title_fontsize=14, loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_dir, 'item_pearson_bars.jpg'), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self, metrics):

        report_lines = ["# LIRA模型评估报告\n"]

        report_lines.append("## 总分级评估结果\n")
        for metric, value in metrics['total_level'].items():
            report_lines.append(f"- {metric}: {value:.4f}")
        report_lines.append("\n")

        report_lines.append("\\n=== 项目级评估 ===")
        if 'item_level' in metrics:
            for item, m in metrics['item_level'].items():
                if isinstance(m, dict):
                    report_lines.append(f"  {item}: RMSE={m.get('rmse', 'N/A'):.4f}, MAE={m.get('mae', 'N/A'):.4f}, R2={m.get('r2', 'N/A'):.4f}, Pearson R={m.get('pearson_r', 'N/A'):.4f}")
        else:
            report_lines.append("  无项目级指标。")

        report_lines.append("\\n=== 总分级评估 ===")
        for metric, value in metrics['total_level'].items():
            report_lines.append(f"- {metric}: {value:.4f}")
        report_lines.append("\n")

        with open(os.path.join(self.output_dir, 'evaluation_report.md'), 'w') as f:
            f.write('\n'.join(report_lines))

