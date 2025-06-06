import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowNet_CNN(nn.Module):
    """轻量级CNN处理flow数据"""
    
    def __init__(self, num_classes=8):
        super().__init__()
        
        # 空间特征提取器 (输入: [7, 8, 6])
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),  # [32, 8, 6]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [64, 8, 6]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [64, 1, 1]
            nn.Flatten()  # [64]
        )
        
        # 全局特征处理器 (输入: [10])
        self.global_processor = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32)
        )
        
        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, spatial_features, global_features):
        spatial_feat = self.spatial_extractor(spatial_features)  # [batch, 64]
        global_feat = self.global_processor(global_features)     # [batch, 32]
        
        combined = torch.cat([spatial_feat, global_feat], dim=1)  # [batch, 96]
        logits = self.classifier(combined)
        return logits

# 可选：添加其他网络架构
class FlowNet_MLP(nn.Module):
    """简单MLP处理flow特征"""
    
    def __init__(self, num_classes=8):
        super().__init__()
        
        # 输入: 7*8*6 + 10 = 346维
        self.network = nn.Sequential(
            nn.Linear(346, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, spatial_features, global_features):
        # 展平空间特征
        spatial_flat = spatial_features.view(spatial_features.size(0), -1)
        # 拼接所有特征
        combined = torch.cat([spatial_flat, global_features], dim=1)
        return self.network(combined) 