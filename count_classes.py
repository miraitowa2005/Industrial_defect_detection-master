import os

# 统计数据集中的唯一类别数量
def count_unique_classes(data_list_path):
    classes = set()
    encodings = ['utf-8', 'gbk', 'gb2312']
    for encoding in encodings:
        try:
            with open(data_list_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _, label_str = line.split(' ', 1)
                        label = int(label_str)
                        classes.add(label)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(f'无法使用以下编码打开文件: {encodings}')
    
    return classes

# 统计训练数据集
print("=== 训练数据集类别统计 ===")
train_classes = count_unique_classes(r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset\train_small.txt')
print(f"训练数据集类别: {sorted(train_classes)}")
print(f"训练数据集类别数量: {len(train_classes)}")

# 统计验证数据集
print("\n=== 验证数据集类别统计 ===")
val_classes = count_unique_classes(r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset\test_small.txt')
print(f"验证数据集类别: {sorted(val_classes)}")
print(f"验证数据集类别数量: {len(val_classes)}")

# 统计完整训练集
print("\n=== 完整训练数据集类别统计 ===")
full_train_classes = count_unique_classes(r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset\train.txt')
print(f"完整训练数据集类别: {sorted(full_train_classes)}")
print(f"完整训练数据集类别数量: {len(full_train_classes)}")
