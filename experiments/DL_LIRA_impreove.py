
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from multiprocessing import cpu_count

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUManager:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.current_gpu = 0
        self.lock = threading.Lock()
        
        logger.info(f"检测到设备: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU数量: {self.num_gpus}")
            for i in range(self.num_gpus):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("使用CPU进行训练")
    
    def get_device(self, model_id: int = None):
        """获取设备，支持多GPU负载均衡"""
        if not torch.cuda.is_available():
            return self.device
        
        with self.lock:
            if model_id is not None:
                gpu_id = model_id % self.num_gpus
            else:
                gpu_id = self.current_gpu
                self.current_gpu = (self.current_gpu + 1) % self.num_gpus
            
            return torch.device(f'cuda:{gpu_id}')


class GatedResidualBlock(nn.Module):
    """门控残差块，用于STFN的SFNet组件"""
    
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.swish = nn.SiLU()  # Swish激活函数
        
    def forward(self, x):
        residual = x
        
        # 因果卷积 + 扩张卷积
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        
        # 门控机制
        gate = torch.sigmoid(self.gate_conv(x))
        x = x * gate
        
        # 残差连接
        return x + self.residual_conv(residual)


class LightweightSTFNLayer1(nn.Module):
    """STFN模型第一层 - 空间-时间特征网络"""
    
    def __init__(self, input_dim: int, num_items: int, device: torch.device):
        super(LightweightSTFNLayer1, self).__init__()
        self.device = device
        self.num_items = num_items
        
        # 特征维度
        self.feature_dim = 128
        self.hidden_dim = 64
        
        self.vqwt_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        
        self.sf_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        
        self.hcpc_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.item_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 32),
                nn.LayerNorm(32), 
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(num_items)
        ])
        
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        
        vqwt_features = self.vqwt_net(x)
        
        sf_features = self.sf_net(vqwt_features)
        
        deep_features = self.hcpc_net(sf_features)
        
        item_predictions = []
        for predictor in self.item_predictors:
            noise = torch.randn_like(deep_features) * 0.01
            pred = predictor(deep_features + noise)
            pred = pred * 5.0
            item_predictions.append(pred)
        
        item_predictions = torch.cat(item_predictions, dim=1)
        
        return item_predictions, deep_features


class LightweightDMPFLayer1(nn.Module):
    def __init__(self, input_dim: int, num_items: int, device: torch.device):
        super(LightweightDMPFLayer1, self).__init__()
        self.device = device
        self.num_items = num_items
        
        self.hidden_dim = 128
        
        self.voiceprint_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.emotion_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.pause_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.energy_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.decoupler = nn.ModuleDict({
            'voiceprint': self._create_decoupler(),
            'emotion': self._create_decoupler(),
            'pause': self._create_decoupler(),
            'energy': self._create_decoupler()
        })
        
        self.gat = self._create_gat()
        
        self.item_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.5), 
                nn.Linear(64, 1),
                nn.Sigmoid() 
            ) for _ in range(num_items)
        ])
        
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _create_decoupler(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        )
    
    def _create_gat(self):
        """创建图注意力网络"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2), 
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        
        voiceprint_feat = self.voiceprint_extractor(x)
        emotion_feat = self.emotion_extractor(x)
        pause_feat = self.pause_extractor(x)
        energy_feat = self.energy_extractor(x)
        
        voiceprint_decoupled = self.decoupler['voiceprint'](voiceprint_feat)
        emotion_decoupled = self.decoupler['emotion'](emotion_feat)
        pause_decoupled = self.decoupler['pause'](pause_feat)
        energy_decoupled = self.decoupler['energy'](energy_feat)
        
        voiceprint_shared = voiceprint_decoupled[:, :self.hidden_dim]
        emotion_shared = emotion_decoupled[:, :self.hidden_dim]
        pause_shared = pause_decoupled[:, :self.hidden_dim]
        energy_shared = energy_decoupled[:, :self.hidden_dim]
        
        shared_features = torch.cat([
            voiceprint_shared, emotion_shared, 
            pause_shared, energy_shared
        ], dim=1)
        fused_features = self.gat(shared_features)
        
        item_predictions = []
        for predictor in self.item_predictors:
            noise = torch.randn_like(fused_features) * 0.01
            pred = predictor(fused_features + noise)
            pred = pred * 5.0
            item_predictions.append(pred)
        
        item_predictions = torch.cat(item_predictions, dim=1)
        
        return item_predictions, fused_features


class LightweightLIRALayer2(nn.Module):
    
    def __init__(self, original_features_dim: int, deep_features_dim: int, 
                 n_items: int, device: torch.device):
        super(LightweightLIRALayer2, self).__init__()
        self.device = device
        self.n_items = n_items
        
        self.n_orig_feat = min(20, original_features_dim)
        self.n_deep_feat = deep_features_dim 
        self.n_item_feat = min(15, n_items) 
        
        self.feature_selector = nn.Sequential(
            nn.Linear(original_features_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, self.n_orig_feat),
            nn.Tanh() 
        )

        n_interactions = min(5, self.n_orig_feat) * min(5, self.n_item_feat)

        corrector_input_dim = (
            self.n_orig_feat +
            self.n_deep_feat +
            self.n_item_feat +
            n_interactions +  
            1
        )
        
        self.residual_corrector = nn.Sequential(
            nn.Linear(corrector_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        
        self.l1_regularization = 1e-5
        
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, original_features: torch.Tensor, deep_features: torch.Tensor,
                item_predictions: torch.Tensor, item_weights: torch.Tensor) -> torch.Tensor:
        
        original_features = original_features.to(self.device)
        deep_features = deep_features.to(self.device)
        item_predictions = item_predictions.to(self.device)
        item_weights = item_weights.to(self.device)
        
        base_pred = torch.sum(item_predictions * item_weights, dim=1, keepdim=True)
        
        feature_weights = self.feature_selector(original_features)
        orig_feat_subset = original_features[:, :self.n_orig_feat] * feature_weights
        
        item_feat_subset = item_predictions[:, :self.n_item_feat]
        
        interactions = []
        for i in range(min(5, self.n_orig_feat)):
            for j in range(min(5, self.n_item_feat)):
                interaction = orig_feat_subset[:, i:i+1] * item_feat_subset[:, j:j+1]
                interactions.append(interaction)
        
        interaction_features = torch.cat(interactions, dim=1)
        
        corrector_input = torch.cat([
            orig_feat_subset,    
            deep_features,       
            item_feat_subset,    
            interaction_features,
            base_pred              
        ], dim=1)
        
        residual_pred = self.residual_corrector(corrector_input)
        
        final_pred = base_pred + residual_pred
        
        return final_pred
    
    def get_l1_loss(self):
        l1_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_regularization * l1_loss


class ParallelExperimentTwo:
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        
        from config import SUBSCALES
        self.subscales = SUBSCALES
        
        self.all_items = []
        for items in self.subscales.values():
            self.all_items.extend(items)
        
        self.gpu_manager = GPUManager()
        
        self.training_config = {
            'layer1_epochs': 150,      
            'layer1_lr': 0.0005,  
            'layer1_batch_size': 128,  
            'layer2_epochs': 200,      
            'layer2_lr': 0.0003,  
            'layer2_batch_size': 256,
            'weight_decay': 5e-3, 
            'max_workers': min(8, cpu_count()),
            'early_stopping_patience': 20,
            'early_stopping_delta': 1e-4,   
        }
        
        self.results = {}
        self.mcw_weights = None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[str]]:
        """加载数据"""
        logger.info("加载真实数据...")
        
        acoustic_path = './dataset/CS-NRAC-E.csv'
        questionnaire_path = './dataset/raw_info.csv'
        
        X_df = pd.read_csv(acoustic_path)
        Y_df = pd.read_csv(questionnaire_path)
        
        common_ids = set(X_df['id']).intersection(set(Y_df['id']))
        X_df = X_df[X_df['id'].isin(common_ids)].set_index('id')
        Y_df = Y_df[Y_df['id'].isin(common_ids)].set_index('id')
        
        common_indices = X_df.index.intersection(Y_df.index)
        X_df = X_df.loc[common_indices].dropna()
        Y_df = Y_df.loc[X_df.index]
        
        if Y_df.index.duplicated().any():
            Y_df = Y_df[~Y_df.index.duplicated(keep='first')]
            X_df = X_df.loc[Y_df.index]
        
        X = X_df.values
        valid_items = [item for item in self.all_items if item in Y_df.columns]
        y_items = Y_df[valid_items].values
        
        y_totals = {}
        for subscale_name, items in self.subscales.items():
            valid_subscale_items = [item for item in items if item in Y_df.columns]
            if valid_subscale_items:
                y_totals[subscale_name] = np.sum(Y_df[valid_subscale_items].values, axis=1)
            else:
                y_totals[subscale_name] = np.zeros(len(Y_df))
        
        logger.info(f"数据加载完成: {len(X)}个样本, {X.shape[1]}维特征, {len(valid_items)}个项目")
        
        return X, y_items, y_totals, valid_items
    
    def train_model_parallel(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """并行训练单个模型配置"""
        try:
            model_type = model_config['model_type']
            subscale_name = model_config['subscale_name']
            device = model_config['device']
            X_train = model_config['X_train'].to(device)
            y_items_train = model_config['y_items_train'].to(device)
            y_total_train = model_config['y_total_train'].to(device)
            X_test = model_config['X_test'].to(device)
            y_items_test = model_config['y_items_test'].to(device)
            y_total_test = model_config['y_total_test'].to(device)
            valid_items = model_config['valid_items']
            mcw_weights = model_config['mcw_weights']
            
            logger.info(f"开始训练 {model_type}-{subscale_name} (设备: {device})")
            
            logger.info(f"数据维度 - X_train: {X_train.shape}, y_items_train: {y_items_train.shape}")
            logger.info(f"真实总分范围 - 训练集: [{y_total_train.min().item():.2f}, {y_total_train.max().item():.2f}]")
            logger.info(f"真实总分范围 - 测试集: [{y_total_test.min().item():.2f}, {y_total_test.max().item():.2f}]")
            logger.info(f"MCW权重统计 - 范围: [{mcw_weights.min():.4f}, {mcw_weights.max():.4f}], 均值: {mcw_weights.mean():.4f}")
            
            num_items = len(valid_items)
            if model_type == 'DMPF':
                layer1_model = LightweightDMPFLayer1(X_train.shape[1], num_items, device)
            else:
                layer1_model = LightweightSTFNLayer1(X_train.shape[1], num_items, device)
            
            layer1_history = self._train_layer1(
                layer1_model, X_train, y_items_train, 
                X_test, y_items_test, device
            )
            
            layer1_model.eval()
            with torch.no_grad():
                train_item_pred, train_deep_feat = layer1_model(X_train)
                test_item_pred, test_deep_feat = layer1_model(X_test)
            
            logger.info(f"第一层项目预测范围 - 测试集: [{test_item_pred.min().item():.3f}, {test_item_pred.max().item():.3f}]")
            logger.info(f"第一层项目预测均值 - 测试集: {test_item_pred.mean().item():.3f}")
            logger.info(f"真实项目得分范围 - 测试集: [{y_items_test.min().item():.3f}, {y_items_test.max().item():.3f}]")
            logger.info(f"真实项目得分均值 - 测试集: {y_items_test.mean().item():.3f}")
            
            subscale_items = self.subscales.get(subscale_name, [])
            subscale_indices = [i for i, item in enumerate(valid_items) if item in subscale_items]
            
            if subscale_indices:
                subscale_weights = torch.FloatTensor(mcw_weights)[subscale_indices]
                
                subscale_weights = subscale_weights / subscale_weights.sum()
                
                original_pred = torch.matmul(
                    test_item_pred[:, subscale_indices], 
                    subscale_weights.to(test_item_pred.device)
                ).cpu().numpy()
                
                scale_factor = y_total_test.mean().cpu().numpy() / (original_pred.mean() + 1e-8)
                original_pred = original_pred * scale_factor
            else:
                original_pred = torch.matmul(
                    test_item_pred, 
                    torch.FloatTensor(mcw_weights).to(test_item_pred.device)
                ).cpu().numpy()
                
                scale_factor = y_total_test.mean().cpu().numpy() / (original_pred.mean() + 1e-8)
                original_pred = original_pred * scale_factor
                
            original_mae = mean_absolute_error(y_total_test.cpu().numpy(), original_pred)
            original_rmse = np.sqrt(mean_squared_error(y_total_test.cpu().numpy(), original_pred))
            
            from sklearn.preprocessing import StandardScaler
            scaler_orig = StandardScaler()
            scaler_deep = StandardScaler()
            
            X_train_scaled = torch.FloatTensor(
                scaler_orig.fit_transform(X_train.cpu().numpy())
            )
            train_deep_feat_scaled = torch.FloatTensor(
                scaler_deep.fit_transform(train_deep_feat.cpu().numpy())
            )
            X_test_scaled = torch.FloatTensor(scaler_orig.transform(X_test.cpu().numpy()))
            test_deep_feat_scaled = torch.FloatTensor(scaler_deep.transform(test_deep_feat.cpu().numpy()))
            
            layer2_model = LightweightLIRALayer2(
                X_train_scaled.shape[1], train_deep_feat_scaled.shape[1], 
                num_items, device
            )
            
            layer2_history = self._train_layer2(
                layer2_model, X_train_scaled, train_deep_feat_scaled,
                train_item_pred, y_total_train, mcw_weights, device
            )
            
            layer2_model.eval()
            with torch.no_grad():
                weights_tensor = torch.FloatTensor(mcw_weights).unsqueeze(0).repeat(X_test.shape[0], 1)
                lira_pred = layer2_model(
                    X_test_scaled, test_deep_feat_scaled,
                    test_item_pred, weights_tensor
                ).squeeze().cpu().numpy()
            
            lira_mae = mean_absolute_error(y_total_test.cpu().numpy(), lira_pred)
            lira_rmse = np.sqrt(mean_squared_error(y_total_test.cpu().numpy(), lira_pred))
            
            item_metrics = {}
            subscale_items = self.subscales.get(subscale_name, [])
            test_item_pred_np = test_item_pred.cpu().numpy()
            y_items_test_np = y_items_test.cpu().numpy()
            
            for i, item_name in enumerate(valid_items):
                if item_name in subscale_items:
                    item_mae = mean_absolute_error(y_items_test_np[:, i], test_item_pred_np[:, i])
                    item_rmse = np.sqrt(mean_squared_error(y_items_test_np[:, i], test_item_pred_np[:, i]))
                    
                    item_metrics[item_name] = {
                        'mae': item_mae,
                        'rmse': item_rmse,
                        'true_mean': np.mean(y_items_test_np[:, i]),
                        'pred_mean': np.mean(test_item_pred_np[:, i])
                    }
            
            results = {
                'architecture': f'{model_type}_{subscale_name}',
                'subscale': subscale_name,
                'model_type': model_type,
                'device': str(device),
                'item_metrics': item_metrics,
                'original_method': {
                    'mae': original_mae,
                    'rmse': original_rmse,
                    'predictions': original_pred
                },
                'lira_method': {
                    'mae': lira_mae,
                    'rmse': lira_rmse,
                    'predictions': lira_pred
                },
                'improvement': {
                    'mae_improvement': (original_mae - lira_mae) / original_mae * 100,
                    'rmse_improvement': (original_rmse - lira_rmse) / original_rmse * 100
                },
                'training_history': {
                    'layer1': layer1_history,
                    'layer2': layer2_history
                },
                'ground_truth': y_total_test.cpu().numpy()
            }
            
            logger.info(f"完成训练 {model_type}-{subscale_name}: MAE改进 {results['improvement']['mae_improvement']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"训练 {model_config['model_type']}-{model_config['subscale_name']} 时出错: {str(e)}")
            return None
    
    def _train_layer1(self, model: nn.Module, X_train: torch.Tensor, y_items_train: torch.Tensor,
                      X_val: torch.Tensor, y_items_val: torch.Tensor, device: torch.device) -> Dict[str, List[float]]:
        
        epochs = self.training_config['layer1_epochs']
        lr = self.training_config['layer1_lr']
        batch_size = self.training_config['layer1_batch_size']
        weight_decay = self.training_config['weight_decay']
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, 
            min_lr=1e-6, verbose=False
        )
        
        train_dataset = TensorDataset(X_train.cpu(), y_items_train.cpu())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                item_pred, _ = model(batch_X)
                loss = criterion(item_pred, batch_y)
                
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            model.eval()
            with torch.no_grad():
                val_item_pred, _ = model(X_val)
                val_loss = criterion(val_item_pred, y_items_val)
                val_losses.append(val_loss.item())
            
            scheduler.step(val_loss.item())
            
            if val_loss.item() < best_val_loss - self.training_config['early_stopping_delta']:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.training_config['early_stopping_patience']:
                logger.info(f"Layer1 早停在epoch {epoch}，最佳验证损失: {best_val_loss:.4f}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
            
            if epoch % 20 == 0:
                logger.info(f"Layer1 Epoch {epoch}: Train {avg_train_loss:.4f}, Val {val_loss.item():.4f}")
        
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def _train_layer2(self, model: LightweightLIRALayer2, X_train: torch.Tensor, deep_feat_train: torch.Tensor,
                      item_pred_train: torch.Tensor, y_total_train: torch.Tensor, 
                      mcw_weights: np.ndarray, device: torch.device) -> Dict[str, List[float]]:
        """训练第二层LIRA模型，添加验证集评估防止过拟合"""
        epochs = self.training_config['layer2_epochs']
        lr = self.training_config['layer2_lr']
        batch_size = self.training_config['layer2_batch_size']
        weight_decay = self.training_config['weight_decay']
        
        # 创建验证集（从训练集中划分20%）
        train_size = int(0.8 * X_train.shape[0])
        indices = torch.randperm(X_train.shape[0])
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train_split = X_train[train_indices]
        deep_feat_train_split = deep_feat_train[train_indices]
        item_pred_train_split = item_pred_train[train_indices]
        y_total_train_split = y_total_train[train_indices]
        
        X_val = X_train[val_indices]
        deep_feat_val = deep_feat_train[val_indices]
        item_pred_val = item_pred_train[val_indices]
        y_total_val = y_total_train[val_indices]
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, 
            min_lr=1e-6, verbose=False
        )
        
        weights_tensor_train = torch.FloatTensor(mcw_weights).unsqueeze(0).repeat(train_size, 1)
        weights_tensor_val = torch.FloatTensor(mcw_weights).unsqueeze(0).repeat(len(val_indices), 1)
        
        train_dataset = TensorDataset(
            X_train_split.cpu(), 
            deep_feat_train_split.cpu(), 
            item_pred_train_split.cpu(), 
            weights_tensor_train.cpu(), 
            y_total_train_split.cpu()
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        train_losses = []
        val_losses = []
        
        # 早停参数
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_orig, batch_deep, batch_items, batch_weights, batch_targets in train_loader:
                batch_orig = batch_orig.to(device)
                batch_deep = batch_deep.to(device)
                batch_items = batch_items.to(device)
                batch_weights = batch_weights.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                
                pred = model(batch_orig, batch_deep, batch_items, batch_weights)
                mse_loss = criterion(pred.squeeze(), batch_targets)
                
                l1_loss = model.get_l1_loss()
                loss = mse_loss + l1_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += mse_loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            model.eval()
            with torch.no_grad():
                val_pred = model(
                    X_val.to(device),
                    deep_feat_val.to(device),
                    item_pred_val.to(device),
                    weights_tensor_val.to(device)
                ).squeeze()
                
                val_loss = criterion(val_pred, y_total_val.to(device))
                val_losses.append(val_loss.item())
            
            scheduler.step(val_loss.item())
            
            if val_loss.item() < best_val_loss - self.training_config['early_stopping_delta']:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.training_config['early_stopping_patience']:
                logger.info(f"Layer2 早停在epoch {epoch}，最佳验证损失: {best_val_loss:.4f}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
            
            if epoch % 40 == 0:
                logger.info(f"Layer2 Epoch {epoch}: Train {avg_train_loss:.4f}, Val {val_loss.item():.4f}")
        
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def run_parallel_experiment(self) -> Dict[str, Any]:
       
        logger.info("开始运行GPU并行实验2")
        
        X, y_items, y_totals, valid_items = self.load_data()
        
        y_totals_array = np.column_stack([y_totals[subscale] for subscale in self.subscales.keys()])
        X_train, X_test, y_items_train, y_items_test, y_totals_train, y_totals_test = train_test_split(
            X, y_items, y_totals_array, test_size=0.2, random_state=42
        )
        
        y_totals_train_dict = {}
        y_totals_test_dict = {}
        for i, subscale in enumerate(self.subscales.keys()):
            y_totals_train_dict[subscale] = y_totals_train[:, i]
            y_totals_test_dict[subscale] = y_totals_test[:, i]
        
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from weight_calculator import WeightCalculator
        
        y_items_df = pd.DataFrame(y_items_train, columns=valid_items)
        weight_calculator = WeightCalculator(y_items_df, subscale_mapping=self.subscales)
        item_weights, subscale_weights = weight_calculator.calculate_weights(calculate_module_controllability=False)
        
        mcw_weights = []
        for item in valid_items:
            mcw_weights.append(item_weights.get(item, 1.0 / len(valid_items)))
        mcw_weights = np.array(mcw_weights)
        
        mcw_weights = mcw_weights / mcw_weights.sum()
        
        logger.info(f"MCW权重计算完成，形状: {mcw_weights.shape}")
        logger.info(f"权重统计 - 最小值: {mcw_weights.min():.4f}, 最大值: {mcw_weights.max():.4f}, 和: {mcw_weights.sum():.4f}")
        
        model_configs = []
        model_id = 0
        
        for subscale_name in self.subscales.keys():
            for model_type in ['DMPF', 'STFN']:
                device = self.gpu_manager.get_device(model_id)
                
                config = {
                    'model_type': model_type,
                    'subscale_name': subscale_name,
                    'device': device,
                    'X_train': torch.FloatTensor(X_train),
                    'y_items_train': torch.FloatTensor(y_items_train),
                    'y_total_train': torch.FloatTensor(y_totals_train_dict[subscale_name]),
                    'X_test': torch.FloatTensor(X_test),
                    'y_items_test': torch.FloatTensor(y_items_test),
                    'y_total_test': torch.FloatTensor(y_totals_test_dict[subscale_name]),
                    'valid_items': valid_items,
                    'mcw_weights': mcw_weights
                }
                
                model_configs.append(config)
                model_id += 1
        
        results = {}
        max_workers = self.training_config['max_workers']
        
        logger.info(f"启动{len(model_configs)}个并行训练任务，最大工作进程数: {max_workers}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
          
            future_to_config = {
                executor.submit(self.train_model_parallel, config): config 
                for config in model_configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result is not None:
                        subscale = result['subscale']
                        model_type = result['model_type']
                        
                        if subscale not in results:
                            results[subscale] = {}
                        results[subscale][model_type] = result
                        
                except Exception as e:
                    logger.error(f"任务失败 {config['model_type']}-{config['subscale_name']}: {str(e)}")
        
        logger.info("所有并行训练任务完成")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        
        for subscale_name, subscale_results in results.items():
            print(f"\n{'='*20} {subscale_name} 量表结果 {'='*20}")
            
            for arch_name, arch_results in subscale_results.items():
                print(f"\n=== {arch_name} 架构 - {subscale_name} ===")
                print(f"训练设备: {arch_results['device']}")
            
                # 总分预测对比
                print(f"\n{subscale_name} 总分预测对比:")
                print("-"*40)
                print(f"{'方法':<15} {'MAE':<10} {'RMSE':<10}")
                print("-"*40)
                print(f"{'原始方法':<15} {arch_results['original_method']['mae']:<10.3f} "
                      f"{arch_results['original_method']['rmse']:<10.3f}")
                print(f"{'LIRA方法':<15} {arch_results['lira_method']['mae']:<10.3f} "
                      f"{arch_results['lira_method']['rmse']:<10.3f}")
                
                print(f"\n性能改进:")
                print("-"*30)
                print(f"MAE改进: {arch_results['improvement']['mae_improvement']:.2f}%")
                print(f"RMSE改进: {arch_results['improvement']['rmse_improvement']:.2f}%")
        
        print(f"\n{'='*20} GPU并行训练性能总结 {'='*20}")
        print("-"*80)
        print(f"{'量表-架构':<15} {'设备':<12} {'原始MAE':<10} {'LIRA MAE':<10} {'MAE改进%':<12}")
        print("-"*80)
        for subscale_name, subscale_results in results.items():
            for arch_name, arch_results in subscale_results.items():
                key = f"{subscale_name}-{arch_name}"
                device = arch_results['device']
                print(f"{key:<15} {device:<12} {arch_results['original_method']['mae']:<10.3f} "
                      f"{arch_results['lira_method']['mae']:<10.3f} "
                      f"{arch_results['improvement']['mae_improvement']:<12.2f}")
    
    def save_results(self, results: Dict[str, Any], save_dir: str = "results/experiment2_parallel"):
        """保存结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(results, os.path.join(save_dir, "parallel_experiment2_results.pkl"))
        
        comparison_data = []
        for subscale_name, subscale_results in results.items():
            for arch_name, arch_results in subscale_results.items():
                comparison_data.append({
                    '量表': subscale_name,
                    '架构': arch_name,
                    '设备': arch_results['device'],
                    '原始MAE': arch_results['original_method']['mae'],
                    '原始RMSE': arch_results['original_method']['rmse'],
                    'LIRA_MAE': arch_results['lira_method']['mae'],
                    'LIRA_RMSE': arch_results['lira_method']['rmse'],
                    'MAE改进%': arch_results['improvement']['mae_improvement'],
                    'RMSE改进%': arch_results['improvement']['rmse_improvement']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(save_dir, "parallel_performance_comparison.csv"), index=False)
        
        logger.info(f"并行实验结果已保存到: {save_dir}")
    
    def run(self):
        """运行完整的并行实验"""
        results = self.run_parallel_experiment()
        self.print_results(results)
        self.save_results(results)
        return results


if __name__ == "__main__":
    # 设置GPU优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    os.makedirs("results", exist_ok=True)
    
    experiment = ParallelExperimentTwo()
    results = experiment.run() 