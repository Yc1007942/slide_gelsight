import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

from multimodal_dataset import MultimodalTactileDataset
from multimodal_network import MultimodalFusionNet

class MultimodalTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建数据集
        self.train_dataset = MultimodalTactileDataset(
            data_root=config['data_root'],
            split='train',
            sequence_length=config['sequence_length'],
            img_size=config['img_size'],
            augment=True
        )
        
        self.test_dataset = MultimodalTactileDataset(
            data_root=config['data_root'],
            split='test',
            sequence_length=config['sequence_length'],
            img_size=config['img_size'],
            augment=False
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # 创建模型
        self.model = MultimodalFusionNet(
            num_classes=config['num_classes'],
            img_hidden=config['img_hidden'],
            flow_hidden=config['flow_hidden']
        ).to(self.device)
        
        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.2
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 训练历史
        self.train_history = {'loss': [], 'acc': []}
        self.test_history = {'loss': [], 'acc': []}
        
        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"./results/multimodal_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)          # [B, 13, 3, 224, 224]
            flow_spatial = batch['flow_spatial'].to(self.device)  # [B, 12, 7, 8, 6]
            flow_global = batch['flow_global'].to(self.device)    # [B, 12, 10]
            labels = batch['label'].to(self.device)           # [B]
            
            # 前向传播
            self.optimizer.zero_grad()
            logits, features = self.model(images, flow_spatial, flow_global)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for batch in pbar:
                images = batch['images'].to(self.device)
                flow_spatial = batch['flow_spatial'].to(self.device)
                flow_global = batch['flow_global'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, features = self.model(images, flow_spatial, flow_global)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = total_loss / len(self.test_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self):
        print(f"开始训练，共 {self.config['epochs']} 轮")
        best_acc = 0
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            
            # 测试
            test_loss, test_acc, test_preds, test_labels = self.test_epoch()
            self.test_history['loss'].append(test_loss)
            self.test_history['acc'].append(test_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_acc': test_acc,
                    'config': self.config
                }, os.path.join(self.save_dir, 'best_model.pth'))
                
                # 保存详细结果
                self.save_detailed_results(test_labels, test_preds, epoch)
            
            # 定期保存训练历史
            if (epoch + 1) % 10 == 0:
                self.save_training_history()
        
        print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")
        return best_acc
    
    def save_detailed_results(self, true_labels, pred_labels, epoch):
        """保存详细的测试结果"""
        # 分类报告
        report = classification_report(true_labels, pred_labels,
                                     target_names=[f'Material_{i+1}' for i in range(self.config['num_classes'])],
                                     output_dict=True)
        
        # 保存为JSON
        with open(os.path.join(self.save_dir, f'classification_report_epoch_{epoch}.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'M{i+1}' for i in range(self.config['num_classes'])],
                   yticklabels=[f'M{i+1}' for i in range(self.config['num_classes'])])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()
    
    def save_training_history(self):
        """保存训练历史"""
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['loss'], label='Train Loss')
        plt.plot(self.test_history['loss'], label='Test Loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_history['acc'], label='Train Acc')
        plt.plot(self.test_history['acc'], label='Test Acc')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()

def main():
    # 配置参数
    config = {
        'data_root': './dataset_split',    # 数据根目录
        'sequence_length': 13,             # 序列长度
        'img_size': 224,                   # 图像尺寸
        'batch_size': 8,                   # 批次大小
        'num_workers': 4,                  # 数据加载进程数
        'num_classes': 10,                 # 类别数
        'img_hidden': 512,                 # 图像LSTM隐藏层维度
        'flow_hidden': 256,                # Flow LSTM隐藏层维度
        'learning_rate': 1e-4,             # 学习率
        'weight_decay': 1e-4,              # 权重衰减
        'epochs': 100,                     # 训练轮数
        'device': 'cuda'                   # 设备
    }
    
    # 保存配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./results/multimodal_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建训练器并开始训练
    trainer = MultimodalTrainer(config)
    best_acc = trainer.train()
    
    print(f"训练完成，最佳准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    main() 