try:
    from flow_processor import FlowFeatureExtractor
    print("✓ flow_processor 导入成功")
except Exception as e:
    print(f"✗ flow_processor 导入失败: {e}")

try:
    from flow_networks import FlowNet_CNN
    print("✓ flow_networks 导入成功")
except Exception as e:
    print(f"✗ flow_networks 导入失败: {e}")

try:
    from multimodal_network import MultimodalFusionNet
    print("✓ multimodal_network 导入成功")
except Exception as e:
    print(f"✗ multimodal_network 导入失败: {e}")

try:
    from multimodal_dataset import MultimodalTactileDataset
    print("✓ multimodal_dataset 导入成功")
except Exception as e:
    print(f"✗ multimodal_dataset 导入失败: {e}")

print("所有导入测试完成") 