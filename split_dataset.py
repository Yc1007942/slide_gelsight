#!/usr/bin/env python3
import os
import glob
import random
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def discover_materials(root_dir: str) -> Dict[int, List[str]]:
    """
    发现所有材料和对应的cycle目录
    返回: {material_id: [cycle_paths]}
    """
    materials = defaultdict(list)
    
    for material_dir in sorted(glob.glob(os.path.join(root_dir, "material_*"))):
        # 从目录名提取材料ID (material_1 -> 1)
        material_id = int(os.path.basename(material_dir).split("_")[1])
        
        # 获取该材料下的所有cycle目录
        cycle_dirs = sorted(glob.glob(os.path.join(material_dir, "cycle_*")))
        materials[material_id] = cycle_dirs
        
        print(f"材料 {material_id}: 发现 {len(cycle_dirs)} 个cycles")
    
    return materials

def split_data(materials: Dict[int, List[str]], train_ratio: float = 0.9, 
               random_seed: int = 42) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    将每个材料的数据按指定比例分为训练集和测试集
    """
    random.seed(random_seed)
    
    train_split = defaultdict(list)
    test_split = defaultdict(list)
    
    for material_id, cycles in materials.items():
        # 随机打乱cycles
        shuffled_cycles = cycles.copy()
        random.shuffle(shuffled_cycles)
        
        # 计算分割点
        num_cycles = len(cycles)
        train_size = int(num_cycles * train_ratio)
        
        # 分割
        train_split[material_id] = shuffled_cycles[:train_size]
        test_split[material_id] = shuffled_cycles[train_size:]
        
        print(f"材料 {material_id}: 训练集 {len(train_split[material_id])} cycles, "
              f"测试集 {len(test_split[material_id])} cycles")
    
    return train_split, test_split

def create_split_structure(train_split: Dict[int, List[str]], 
                          test_split: Dict[int, List[str]], 
                          output_dir: str,
                          mode: str = "symlink"):
    """
    创建训练集和测试集的目录结构
    mode: "copy" (复制文件), "symlink" (创建符号链接), "list" (只生成文件列表)
    """
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    
    # 创建输出目录
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    def process_split(split_data: Dict[int, List[str]], split_dir: Path, split_name: str):
        for material_id, cycles in split_data.items():
            material_dir = split_dir / f"material_{material_id}"
            material_dir.mkdir(exist_ok=True)
            
            for cycle_path in cycles:
                cycle_name = os.path.basename(cycle_path)
                target_path = material_dir / cycle_name
                
                if mode == "copy":
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(cycle_path, target_path)
                    
                elif mode == "symlink":
                    if target_path.exists():
                        target_path.unlink()
                    target_path.symlink_to(os.path.abspath(cycle_path))
                    
                # mode == "list" 的情况在后面处理
        
        print(f"{split_name}集目录结构创建完成: {split_dir}")
    
    if mode in ["copy", "symlink"]:
        process_split(train_split, train_dir, "训练")
        process_split(test_split, test_dir, "测试")
    
    # 无论什么模式都生成文件列表
    save_file_lists(train_split, test_split, output_path)

def save_file_lists(train_split: Dict[int, List[str]], 
                   test_split: Dict[int, List[str]], 
                   output_dir: Path):
    """
    保存训练集和测试集的文件列表
    """
    # 生成训练集文件列表
    train_files = []
    for material_id, cycles in train_split.items():
        for cycle_path in cycles:
            # 获取该cycle下的所有frame文件
            frame_files = sorted(glob.glob(os.path.join(cycle_path, "frame_*.png")))
            for frame_file in frame_files:
                train_files.append({
                    "path": frame_file,
                    "material_id": material_id,
                    "cycle": os.path.basename(cycle_path),
                    "frame": os.path.basename(frame_file)
                })
    
    # 生成测试集文件列表
    test_files = []
    for material_id, cycles in test_split.items():
        for cycle_path in cycles:
            frame_files = sorted(glob.glob(os.path.join(cycle_path, "frame_*.png")))
            for frame_file in frame_files:
                test_files.append({
                    "path": frame_file,
                    "material_id": material_id,
                    "cycle": os.path.basename(cycle_path),
                    "frame": os.path.basename(frame_file)
                })
    
    # 保存为JSON文件
    with open(output_dir / "train_list.json", "w", encoding="utf-8") as f:
        json.dump(train_files, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "test_list.json", "w", encoding="utf-8") as f:
        json.dump(test_files, f, indent=2, ensure_ascii=False)
    
    # 保存简单的路径列表
    with open(output_dir / "train_paths.txt", "w") as f:
        for item in train_files:
            f.write(f"{item['path']}\t{item['material_id']}\n")
    
    with open(output_dir / "test_paths.txt", "w") as f:
        for item in test_files:
            f.write(f"{item['path']}\t{item['material_id']}\n")
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"测试集: {len(test_files)} 个文件")
    print(f"文件列表已保存到: {output_dir}")

def print_statistics(materials: Dict[int, List[str]], 
                    train_split: Dict[int, List[str]], 
                    test_split: Dict[int, List[str]]):
    """
    打印数据集统计信息
    """
    print("\n" + "="*50)
    print("数据集分割统计")
    print("="*50)
    
    total_train_cycles = sum(len(cycles) for cycles in train_split.values())
    total_test_cycles = sum(len(cycles) for cycles in test_split.values())
    total_cycles = sum(len(cycles) for cycles in materials.values())
    
    print(f"总体统计:")
    print(f"  总cycles: {total_cycles}")
    print(f"  训练集cycles: {total_train_cycles} ({total_train_cycles/total_cycles*100:.1f}%)")
    print(f"  测试集cycles: {total_test_cycles} ({total_test_cycles/total_cycles*100:.1f}%)")
    
    print(f"\n按材料统计:")
    for material_id in sorted(materials.keys()):
        total = len(materials[material_id])
        train = len(train_split[material_id])
        test = len(test_split[material_id])
        print(f"  材料 {material_id}: 总计{total:3d} | 训练{train:3d} | 测试{test:2d} | 比例 {train/total*100:.1f}%:{test/total*100:.1f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="将GelSight数据集分割为训练集和测试集")
    parser.add_argument("--root", default=".", help="数据根目录")
    parser.add_argument("--output", default="./dataset_split", help="输出目录")
    parser.add_argument("--ratio", type=float, default=0.9, help="训练集比例 (默认0.9)")
    parser.add_argument("--mode", choices=["copy", "symlink", "list"], default="symlink", 
                       help="输出模式: copy(复制文件), symlink(符号链接), list(仅列表)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print(f"开始处理数据集...")
    print(f"数据根目录: {args.root}")
    print(f"输出目录: {args.output}")
    print(f"训练集比例: {args.ratio}")
    print(f"输出模式: {args.mode}")
    print(f"随机种子: {args.seed}")
    
    # 发现所有材料
    materials = discover_materials(args.root)
    
    if not materials:
        print("错误: 未找到任何材料目录!")
        return
    
    # 分割数据
    train_split, test_split = split_data(materials, args.ratio, args.seed)
    
    # 打印统计信息
    print_statistics(materials, train_split, test_split)
    
    # 创建输出结构
    create_split_structure(train_split, test_split, args.output, args.mode)
    
    print(f"\n数据集分割完成! 输出保存在: {args.output}")

if __name__ == "__main__":
    main()
