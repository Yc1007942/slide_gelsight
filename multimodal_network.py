import torch
import torch.nn as nn
import torchvision.models as models
from flow_networks import FlowNet_CNN

class ImageSequenceEncoder(nn.Module):
    """图像序列编码器 - ResNet + LSTM"""
    
    def __init__(self, hidden_dim=512, num_layers=2):
        super().__init__()
        
        # 使用预训练ResNet18作为特征提取器
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的FC层
        
        # 冻结前几层
        for param in list(self.feature_extractor.parameters())[:20]:
            param.requires_grad = False
        
        # LSTM处理时序特征
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet18的特征维度
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_dim = hidden_dim * 2  # bidirectional
        
    def forward(self, image_sequence):
        batch_size, seq_len, c, h, w = image_sequence.shape
        
        # 展平batch和sequence维度
        images = image_sequence.view(batch_size * seq_len, c, h, w)
        
        # 提取图像特征
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.feature_extractor(images)  # [batch*seq, 512, 1, 1]
        
        features = features.view(batch_size, seq_len, -1)  # [batch, seq, 512]
        
        # LSTM处理时序
        lstm_out, (hidden, cell) = self.lstm(features)  # [batch, seq, hidden*2]
        
        # 使用最后一个时间步的输出
        sequence_feature = lstm_out[:, -1, :]  # [batch, hidden*2]
        
        return sequence_feature

class FlowSequenceEncoder(nn.Module):
    """Flow序列编码器 - CNN + LSTM"""
    
    def __init__(self, hidden_dim=256, num_layers=2):
        super().__init__()
        
        # Flow特征提取器
        self.flow_cnn = FlowNet_CNN(num_classes=128)  # 输出128维特征而不是分类
        # 修改最后的分类器为特征提取器
        self.flow_cnn.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
        
        # LSTM处理时序
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_dim = hidden_dim * 2
        
    def forward(self, flow_spatial_sequence, flow_global_sequence):
        batch_size, seq_len = flow_spatial_sequence.shape[:2]
        
        # 处理每个时间步的flow数据
        flow_features = []
        for t in range(seq_len):
            spatial_t = flow_spatial_sequence[:, t]  # [batch, 7, 8, 6]
            global_t = flow_global_sequence[:, t]    # [batch, 10]
            
            flow_feat_t = self.flow_cnn(spatial_t, global_t)  # [batch, 128]
            flow_features.append(flow_feat_t)
        
        flow_features = torch.stack(flow_features, dim=1)  # [batch, seq, 128]
        
        # LSTM处理时序
        lstm_out, (hidden, cell) = self.lstm(flow_features)
        sequence_feature = lstm_out[:, -1, :]  # [batch, hidden*2]
        
        return sequence_feature

class MultimodalFusionNet(nn.Module):
    """多模态融合网络"""
    
    def __init__(self, num_classes=8, img_hidden=512, flow_hidden=256):
        super().__init__()
        
        # 两个模态的编码器
        self.image_encoder = ImageSequenceEncoder(hidden_dim=img_hidden)
        self.flow_encoder = FlowSequenceEncoder(hidden_dim=flow_hidden)
        
        # 注意力融合
        self.img_attention = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.flow_attention = nn.Sequential(
            nn.Linear(self.flow_encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 最终分类器
        fusion_dim = self.image_encoder.output_dim + self.flow_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, flow_spatial, flow_global):
        # 编码两个模态
        img_features = self.image_encoder(images)        # [batch, img_dim]
        flow_features = self.flow_encoder(flow_spatial, flow_global)  # [batch, flow_dim]
        
        # 计算注意力权重
        img_weight = self.img_attention(img_features)    # [batch, 1]
        flow_weight = self.flow_attention(flow_features) # [batch, 1]
        
        # 注意力加权
        weighted_img = img_features * img_weight
        weighted_flow = flow_features * flow_weight
        
        # 融合特征
        fused_features = torch.cat([weighted_img, weighted_flow], dim=1)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, {
            'img_features': img_features,
            'flow_features': flow_features,
            'img_weight': img_weight,
            'flow_weight': flow_weight
        } 