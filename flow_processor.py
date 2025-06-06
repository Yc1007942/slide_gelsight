import numpy as np
import torch
import torch.nn as nn
import json
from typing import Tuple, Dict

class FlowFeatureExtractor:
    """Flow数据特征提取器"""
    
    def __init__(self):
        self.grid_h, self.grid_w = 8, 6  # 点阵尺寸
    
    def load_flow_data(self, flow_path: str) -> np.ndarray:
        """加载并解析flow数据"""
        with open(flow_path, 'r') as f:
            data = json.load(f)
        
        # 转换为numpy数组: [5, 8, 6] -> [ox, oy, cx, cy, acc]
        flow_array = np.array(data)  # shape: [5, 8, 6]
        return flow_array
    
    def extract_displacement_features(self, flow_data: np.ndarray) -> Dict[str, np.ndarray]:
        """提取位移相关特征"""
        ox, oy, cx, cy, acc = flow_data  # 每个都是 [8, 6]
        
        # 1. 位移向量
        dx = cx - ox  # x方向位移
        dy = cy - oy  # y方向位移
        
        # 2. 位移幅度和方向
        magnitude = np.sqrt(dx**2 + dy**2)  # 位移幅度
        angle = np.arctan2(dy, dx)  # 位移角度
        
        # 3. 置信度加权的位移
        weighted_dx = dx * acc
        weighted_dy = dy * acc
        
        return {
            'displacement_x': dx,           # [8, 6] x方向位移
            'displacement_y': dy,           # [8, 6] y方向位移  
            'magnitude': magnitude,         # [8, 6] 位移幅度
            'angle': angle,                 # [8, 6] 位移角度
            'confidence': acc,              # [8, 6] 置信度
            'weighted_dx': weighted_dx,     # [8, 6] 置信度加权x位移
            'weighted_dy': weighted_dy,     # [8, 6] 置信度加权y位移
        }
    
    def extract_deformation_features(self, displacement_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """提取形变特征（全局标量特征）"""
        dx = displacement_features['displacement_x']
        dy = displacement_features['displacement_y']
        acc = displacement_features['confidence']
        
        # 1. 全局统计特征
        mean_dx = np.mean(dx * acc) / np.mean(acc)  # 置信度加权平均位移
        mean_dy = np.mean(dy * acc) / np.mean(acc)
        std_dx = np.std(dx)  # 位移标准差
        std_dy = np.std(dy)
        
        # 2. 形变强度
        deformation_intensity = np.mean(displacement_features['magnitude'])
        max_deformation = np.max(displacement_features['magnitude'])
        
        # 3. 形变不均匀性
        deformation_uniformity = np.std(displacement_features['magnitude'])
        
        # 4. 方向一致性
        angles = displacement_features['angle']
        # 计算角度的圆形标准差（考虑角度的周期性）
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        angle_consistency = 1.0 / (1.0 + np.std(np.mod(angles - mean_angle + np.pi, 2*np.pi) - np.pi))
        
        # 5. 边界效应（边缘vs中心的形变差异）
        center_mask = np.zeros((8, 6), dtype=bool)
        center_mask[2:6, 1:5] = True  # 中心区域
        center_deformation = np.mean(displacement_features['magnitude'][center_mask])
        edge_deformation = np.mean(displacement_features['magnitude'][~center_mask])
        boundary_effect = abs(center_deformation - edge_deformation)
        
        return {
            'mean_displacement_x': mean_dx,
            'mean_displacement_y': mean_dy,
            'std_displacement_x': std_dx,
            'std_displacement_y': std_dy,
            'deformation_intensity': deformation_intensity,
            'max_deformation': max_deformation,
            'deformation_uniformity': deformation_uniformity,
            'angle_consistency': angle_consistency,
            'boundary_effect': boundary_effect,
            'mean_confidence': np.mean(acc)
        }
    
    def compute_spatial_gradients(self, displacement_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """计算空间梯度特征"""
        dx = displacement_features['displacement_x']
        dy = displacement_features['displacement_y']
        
        # 计算梯度（简单差分）
        grad_dx_x = np.gradient(dx, axis=1)  # x方向位移的x梯度
        grad_dx_y = np.gradient(dx, axis=0)  # x方向位移的y梯度
        grad_dy_x = np.gradient(dy, axis=1)  # y方向位移的x梯度
        grad_dy_y = np.gradient(dy, axis=0)  # y方向位移的y梯度
        
        # 应变张量相关
        strain_xx = grad_dx_x  # 正应变
        strain_yy = grad_dy_y  # 正应变
        strain_xy = 0.5 * (grad_dx_y + grad_dy_x)  # 剪切应变
        
        return {
            'strain_xx': strain_xx,  # [8, 6]
            'strain_yy': strain_yy,  # [8, 6]
            'strain_xy': strain_xy,  # [8, 6]
        }
    
    def process_flow_to_features(self, flow_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        将flow数据处理成网络输入特征
        返回: (spatial_features, global_features)
        """
        # 加载数据
        flow_data = self.load_flow_data(flow_path)
        
        # 提取位移特征
        disp_features = self.extract_displacement_features(flow_data)
        
        # 提取形变特征
        deform_features = self.extract_deformation_features(disp_features)
        
        # 提取梯度特征
        gradient_features = self.compute_spatial_gradients(disp_features)
        
        # 组装空间特征 [channels, height, width]
        spatial_features = np.stack([
            disp_features['weighted_dx'],      # 置信度加权x位移
            disp_features['weighted_dy'],      # 置信度加权y位移
            disp_features['magnitude'],        # 位移幅度
            disp_features['confidence'],       # 置信度
            gradient_features['strain_xx'],    # x方向正应变
            gradient_features['strain_yy'],    # y方向正应变
            gradient_features['strain_xy'],    # 剪切应变
        ], axis=0)  # [7, 8, 6]
        
        # 组装全局特征
        global_features = np.array([
            deform_features['mean_displacement_x'],
            deform_features['mean_displacement_y'],
            deform_features['std_displacement_x'],
            deform_features['std_displacement_y'],
            deform_features['deformation_intensity'],
            deform_features['max_deformation'],
            deform_features['deformation_uniformity'],
            deform_features['angle_consistency'],
            deform_features['boundary_effect'],
            deform_features['mean_confidence']
        ])  # [10]
        
        return spatial_features, global_features 