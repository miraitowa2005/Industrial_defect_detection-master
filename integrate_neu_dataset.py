import os
import shutil
import random

# NEU-DET数据集路径
NEU_DET_PATH = r"d:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\NEU-DET"
# 项目数据集路径
PROJECT_DATASET_PATH = r"d:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset"
# 类别映射关系
CLASS_MAPPING = {
    "crazing": "2_crack",      # 裂纹
    "inclusion": "5_foreign_matter",  # 夹杂
    "patches": "7_lf",         # 斑块
    "pitted_surface": "5_pinhole",  # 麻点表面
    "rolled-in_scale": "6_burr",  # 氧化铁皮压入
    "scratches": "0_scratch"   # 划痕
}

def copy_images(source_dir, target_dir, class_name, target_class_name):
    """将一个类别的图像从源目录复制到目标目录"""
    source_class_dir = os.path.join(source_dir, class_name)
    target_class_dir = os.path.join(target_dir, target_class_name)
    
    # 确保目标目录存在
    os.makedirs(target_class_dir, exist_ok=True)
    
    # 获取源目录中的所有图像文件
    image_files = [f for f in os.listdir(source_class_dir) if f.endswith(".jpg")]
    
    # 复制图像文件
    copied_files = []
    for image_file in image_files:
        source_path = os.path.join(source_class_dir, image_file)
        target_path = os.path.join(target_class_dir, f"{target_class_name}_{len(copied_files)}.jpg")
        shutil.copy2(source_path, target_path)
        copied_files.append(target_path)
        
        # 打印进度
        if len(copied_files) % 50 == 0:
            print(f"已复制 {len(copied_files)} 张 {class_name} 图像到 {target_class_name}")
    
    return copied_files

def update_data_lists(dataset_path, train_ratio=0.8):
    """更新数据列表文件"""
    # 获取所有类别目录
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".") and not d == "NEU-DET"]
    
    # 收集所有图像文件
    all_images = []
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_path, class_dir)
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_path) if f.endswith(".jpg")]
        all_images.extend(image_files)
    
    # 打乱图像顺序
    random.shuffle(all_images)
    
    # 分割训练集和测试集
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]
    
    # 写入训练集列表
    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        for image_path in train_images:
            # 提取类别ID（目录名中的数字部分）
            class_id = image_path.split(os.sep)[0].split("_")[0]
            f.write(f"{image_path} {class_id}\n")
    
    # 写入测试集列表
    with open(os.path.join(dataset_path, "test.txt"), "w") as f:
        for image_path in test_images:
            # 提取类别ID（目录名中的数字部分）
            class_id = image_path.split(os.sep)[0].split("_")[0]
            f.write(f"{image_path} {class_id}\n")
    
    # 写入小型训练集和测试集（用于快速测试）
    small_train_size = min(500, len(train_images))
    small_test_size = min(100, len(test_images))
    
    with open(os.path.join(dataset_path, "train_small.txt"), "w") as f:
        for image_path in train_images[:small_train_size]:
            class_id = image_path.split(os.sep)[0].split("_")[0]
            f.write(f"{image_path} {class_id}\n")
    
    with open(os.path.join(dataset_path, "test_small.txt"), "w") as f:
        for image_path in test_images[:small_test_size]:
            class_id = image_path.split(os.sep)[0].split("_")[0]
            f.write(f"{image_path} {class_id}\n")
    
    print(f"\n数据列表更新完成：")
    print(f"- 训练集：{len(train_images)} 张图像")
    print(f"- 测试集：{len(test_images)} 张图像")
    print(f"- 小型训练集：{small_train_size} 张图像")
    print(f"- 小型测试集：{small_test_size} 张图像")

def main():
    print("=== 整合NEU-DET数据集到项目中 ===")
    
    # 复制训练集图像
    print("\n1. 复制训练集图像...")
    train_source_path = os.path.join(NEU_DET_PATH, "train", "images")
    all_copied_files = []
    
    for neu_class, project_class in CLASS_MAPPING.items():
        print(f"\n正在复制 {neu_class} -> {project_class}...")
        copied_files = copy_images(train_source_path, PROJECT_DATASET_PATH, neu_class, project_class)
        all_copied_files.extend(copied_files)
    
    # 复制验证集图像
    print("\n2. 复制验证集图像...")
    val_source_path = os.path.join(NEU_DET_PATH, "validation", "images")
    if os.path.exists(val_source_path):
        for neu_class, project_class in CLASS_MAPPING.items():
            print(f"\n正在复制验证集 {neu_class} -> {project_class}...")
            copied_files = copy_images(val_source_path, PROJECT_DATASET_PATH, neu_class, project_class)
            all_copied_files.extend(copied_files)
    
    print(f"\n✅ 图像复制完成，共复制 {len(all_copied_files)} 张图像")
    
    # 更新数据列表文件
    print("\n3. 更新数据列表文件...")
    update_data_lists(PROJECT_DATASET_PATH)
    
    print("\n✅ NEU-DET数据集整合完成！")

if __name__ == "__main__":
    main()