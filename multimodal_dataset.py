import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json
from flow_processor import FlowFeatureExtractor

class MultimodalTactileDataset(Dataset):
    """多模态触觉数据集"""
    
    def __init__(self, data_root, split='train', sequence_length=13, 
                 img_size=224, augment=True):
        self.data_root = data_root
        self.split = split
        self.seq_len = sequence_length
        self.augment = augment and split == 'train'
        
        # 图像预处理
        if self.augment:
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Flow特征提取器
        self.flow_extractor = FlowFeatureExtractor()
        
        # 加载数据路径
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """加载所有样本路径"""
        samples = []
        split_dir = os.path.join(self.data_root, self.split)
        
        for material_dir in sorted(glob.glob(os.path.join(split_dir, "material_*"))):
            material_id = int(os.path.basename(material_dir).split("_")[1]) - 1  # 0-7
            
            for cycle_dir in sorted(glob.glob(os.path.join(material_dir, "cycle_*"))):
                # 检查是否有完整的序列
                frame_files = sorted(glob.glob(os.path.join(cycle_dir, "frame_*.png")))
                flow_files = sorted(glob.glob(os.path.join(cycle_dir, "flow_*.json")))
                
                if len(frame_files) >= self.seq_len and len(flow_files) >= (self.seq_len - 1):
                    samples.append({
                        'cycle_path': cycle_dir,
                        'material_id': material_id,
                        'frame_files': frame_files[:self.seq_len],
                        'flow_files': flow_files[:self.seq_len-1]  # flow比frame少1个
                    })
        
        print(f"{self.split}集: 加载了 {len(samples)} 个样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像序列
        image_sequence = []
        for frame_path in sample['frame_files']:
            img = Image.open(frame_path).convert('RGB')
            img_tensor = self.img_transform(img)
            image_sequence.append(img_tensor)
        image_sequence = torch.stack(image_sequence)  # [T, C, H, W]
        
        # 加载flow序列
        flow_spatial_sequence = []
        flow_global_sequence = []
        
        for i, flow_path in enumerate(sample['flow_files']):
            try:
                spatial_feat, global_feat = self.flow_extractor.process_flow_to_features(flow_path)
                flow_spatial_sequence.append(torch.from_numpy(spatial_feat).float())
                flow_global_sequence.append(torch.from_numpy(global_feat).float())
            except Exception as e:
                print(f"Warning: Failed to load flow {flow_path}: {e}")
                # 使用零填充
                flow_spatial_sequence.append(torch.zeros(7, 8, 6))
                flow_global_sequence.append(torch.zeros(10))
        
        # 如果flow序列长度不够，用最后一个重复填充
        while len(flow_spatial_sequence) < self.seq_len - 1:
            flow_spatial_sequence.append(flow_spatial_sequence[-1])
            flow_global_sequence.append(flow_global_sequence[-1])
        
        flow_spatial_sequence = torch.stack(flow_spatial_sequence)  # [T-1, 7, 8, 6]
        flow_global_sequence = torch.stack(flow_global_sequence)    # [T-1, 10]
        
        return {
            'images': image_sequence,                    # [13, 3, 224, 224]
            'flow_spatial': flow_spatial_sequence,       # [12, 7, 8, 6]
            'flow_global': flow_global_sequence,         # [12, 10]
            'label': sample['material_id'],              # int
            'cycle_path': sample['cycle_path']
        } 