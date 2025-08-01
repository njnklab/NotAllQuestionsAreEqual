#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network analysis and weight calculation module
"""

import numpy as np
import pandas as pd
import networkx as nx
import random
import itertools
from scipy import stats
import datetime
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from config import RANDOM_SEED, SUBSCALES, GRAPHICAL_LASSO_CV, GRAPHICAL_LASSO_ALPHA, MDS_ITERATIONS
import community.community_louvain as louvain


class WeightCalculator:
    """Weight calculation class"""

    def __init__(self, Y_train, subscale_mapping=SUBSCALES):
        """Initialize weight calculator"""
        self.Y_train = Y_train
        self.subscale_mapping = subscale_mapping
        self.graph = None
        self.valid_items = []
        self.all_dom_sets = []
        self.communities = None
        self.item_weights = None
        self.subscale_weights = None
        self.module_controllability = None

    def calculate_weights(self, calculate_module_controllability=True):
        """Calculate weights"""
        print("Calculating item weights...")
        start_time = datetime.datetime.now()
        
        self.valid_items = []
        for items in self.subscale_mapping.values():
            self.valid_items.extend([item for item in items if item in self.Y_train.columns])

        corr_data = self.Y_train[self.valid_items]

        self.graph = self._construct_network(corr_data, self.valid_items)

        if louvain is not None and calculate_module_controllability:
            print("Performing community detection...")
            self.communities = louvain.best_partition(self.graph)
            num_communities = max(self.communities.values()) + 1
            print(f"Detected {num_communities} communities")

        print(f"Calculating minimum dominating set (repeat {MDS_ITERATIONS} times)...")
        self.all_dom_sets = self._greedy_minimum_dominating_set(self.graph, MDS_ITERATIONS)
        
        self.item_weights = self._dominating_frequency(self.all_dom_sets, self.graph)
        print("Control frequency (CF) values calculation completed")

        print("Calculating subscale weights...")
        self.subscale_weights = {}

        for subscale, items in self.subscale_mapping.items():
            valid_subscale_items = [item for item in items if item in self.valid_items]
            if valid_subscale_items:
                self.subscale_weights[subscale] = np.mean([self.item_weights[item] for item in valid_subscale_items])
            else:
                self.subscale_weights[subscale] = 0.0

        if louvain is not None and calculate_module_controllability and self.communities is not None:
            self.module_controllability = self._calculate_module_controllability()

        self.item_weights = pd.Series(self.item_weights)
        self.subscale_weights = pd.Series(self.subscale_weights)

        end_time = datetime.datetime.now()
        print(f"Weight calculation completed, time: {(end_time - start_time).total_seconds():.2f} seconds")
        return self.item_weights, self.subscale_weights

    def _construct_network(self, data, columns):
        """Construct network using Graphical Lasso"""
        corr_matrix = data.corr()

        print("Estimating sparse inverse covariance matrix using Graphical Lasso...")
        if GRAPHICAL_LASSO_CV:
            model = GraphicalLassoCV(cv=5)
        else:
            model = GraphicalLasso(alpha=GRAPHICAL_LASSO_ALPHA, random_state=RANDOM_SEED)
        model.fit(data)

        precision_matrix = pd.DataFrame(model.precision_, index=columns, columns=columns)
        
        diag_sqrt = np.sqrt(np.diag(model.precision_))
        partial_corr_matrix = -model.precision_ / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(partial_corr_matrix, 1)
        
        processed_matrix = self._matrix_preprocess(partial_corr_matrix)
        
        # 构建条件依赖网络图
        print("Building conditional dependency network...")
        # 使用带有有效项目标签的图
        graph = nx.Graph()
        for i, item1 in enumerate(columns):
            graph.add_node(item1)
            for j, item2 in enumerate(columns):
                if i < j and processed_matrix[i, j] != 0:
                    graph.add_edge(item1, item2, weight=processed_matrix[i, j])
                    
        return graph

    def _matrix_preprocess(self, matrix):
        """Preprocess matrix: remove diagonal and take absolute value"""
        number_of_nodes = matrix.shape[1]
        matrix_result = matrix.copy()
        # 去对角线
        for i in range(0, number_of_nodes):
            matrix_result[i, i] = 0
        # 取绝对值
        matrix_result = abs(matrix_result)
        return matrix_result

    def _greedy_minimum_dominating_set(self, graph, times):
        """Approximate minimum dominating set using greedy algorithm"""
        min_dominating_set = []

        for time in range(times):
            graph_copy = graph.copy()
            dominating_set = []

            while graph_copy.nodes():
                # 随机选择一个节点
                node = random.choice(list(graph_copy.nodes()))
                dominating_set.append(node)
                
                # 移除该节点及其邻居
                remove_list = [node]
                for neighbor in graph_copy.neighbors(node):
                    remove_list.append(neighbor)

                for node_to_remove in remove_list:
                    if graph_copy.has_node(node_to_remove):
                        graph_copy.remove_node(node_to_remove)

            dominating_set = set(dominating_set)
            
            # 更新最小支配集列表
            if len(min_dominating_set) == 0:
                min_dominating_set.append(dominating_set)
            elif len(min_dominating_set[0]) == len(dominating_set) and dominating_set not in min_dominating_set:
                min_dominating_set.append(dominating_set)
            elif len(min_dominating_set[0]) > len(dominating_set):
                min_dominating_set.clear()
                min_dominating_set.append(dominating_set)
                
            if (time + 1) % 100 == 0:
                print(f"Completed {time + 1} iterations, current minimum dominating set size: {len(min_dominating_set[0])}, count: {len(min_dominating_set)}")

        return min_dominating_set

    def _dominating_frequency(self, all_dom_sets, graph):
        """Calculate the frequency of nodes appearing in minimum dominating sets"""
        num_dom_set = len(all_dom_sets)
        
        # 初始化频率计数
        node_list = list(graph.nodes())
        as_dom_node_count = {node: 0 for node in node_list}
        
        # 统计每个节点出现在最小支配集中的次数
        for min_dom_set in all_dom_sets:
            for dom_node in min_dom_set:
                as_dom_node_count[dom_node] = as_dom_node_count[dom_node] + 1

        # 计算频率
        for node in as_dom_node_count:
            as_dom_node_count[node] = as_dom_node_count[node] / num_dom_set
            
        return as_dom_node_count
        
    def _calculate_module_controllability(self):
        """Calculate module controllability"""
        if louvain is None or self.communities is None:
            print("Warning: Community detection results not available, cannot calculate module controllability")
            return {}
            
        # 获取社区数量
        number_of_communities = max(self.communities.values()) + 1
        
        # 按社区分组节点
        module = {i: [] for i in range(number_of_communities)}
        for node, community_index in self.communities.items():
            module[community_index].append(node)
        
        # 初始化结果字典
        average_module_controllability_result = {f"{source}_{target}": 0 
                                                for source in module 
                                                for target in module}
        
        # 对每个最小支配集计算模块可控性
        for min_dom_set in self.all_dom_sets:
            # 计算每个支配节点的控制区域（该节点及其邻居）
            dominated_area = {dom_node: set(self.graph.neighbors(dom_node)).union({dom_node}) 
                             for dom_node in min_dom_set}
            
            # 计算每个模块的控制区域（模块内所有支配节点的控制区域的并集）
            modules_control_area = {}
            for module_index, node_in_module in module.items():
                # 获取该模块内的支配节点
                dom_nodes_in_module = [node for node in node_in_module if node in min_dom_set]
                
                # 如果该模块内有支配节点，计算控制区域
                if dom_nodes_in_module:
                    control_areas = [dominated_area[node] for node in dom_nodes_in_module]
                    modules_control_area[module_index] = set().union(*control_areas)
                else:
                    modules_control_area[module_index] = set()
            
            # 计算模块间的控制能力
            temp_module_controllability_result = {}
            for module_source, control_area in modules_control_area.items():
                for module_target, target_module_area in module.items():
                    # 计算源模块对目标模块的控制能力
                    intersection = control_area.intersection(set(target_module_area))
                    controllability = len(intersection) / len(target_module_area) if target_module_area else 0
                    temp_module_controllability_result[f"{module_source}_{module_target}"] = controllability
                    
                    # 累加到平均结果
                    average_module_controllability_result[f"{module_source}_{module_target}"] += controllability
        
        # 计算平均模块可控性
        for key in average_module_controllability_result:
            average_module_controllability_result[key] /= len(self.all_dom_sets)
            
        return average_module_controllability_result
        
    def get_community_membership(self):
        """Get community membership of nodes"""
        return self.communities
        
    def get_module_controllability(self):
        """Get module controllability matrix"""
        return self.module_controllability
