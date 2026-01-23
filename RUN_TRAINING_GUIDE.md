# 手动运行训练模型指南

本指南将详细介绍如何手动运行工业缺陷检测模型的训练过程。

## 1. 环境准备

### 1.1 检查Python环境
确保您已安装Python 3.6或更高版本，并已配置好PyTorch环境。

```bash
# 检查Python版本
python --version

# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"
```

### 1.2 安装依赖
项目需要以下主要依赖：
- PyTorch
- torchvision
- numpy
- PIL (Pillow)

使用pip安装：

```bash
pip install torch torchvision numpy pillow
```

## 2. 配置训练参数

所有训练参数都在`config.py`文件中定义。您可以根据需要修改以下参数：

```python
class Args:
    data_root = 'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset'  # 数据集根路径
    train_list = 'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset\train_small.txt'  # 训练集列表
    val_list = 'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset\test_small.txt'  # 验证集列表
    arch = 'resnet50'  # 网络架构
    num_classes = 9  # 类别数
    batch_size = 32  # 批量大小
    lr = 0.001  # 学习率
    epoch = 50  # 训练轮数
    # 其他参数...
```

### 2.1 常用参数说明
- **arch**：网络架构，支持resnet50、se_resnet50、efficientnet_b0等
- **batch_size**：根据GPU显存大小调整，显存不足时可减小
- **lr**：学习率，初始值建议0.001
- **epoch**：训练轮数，默认50轮

## 3. 运行训练命令

### 3.1 直接运行（使用config.py配置）

```bash
# 在项目根目录下运行
python train.py
```

### 3.2 使用自定义参数运行

如果您希望不修改config.py而直接通过命令行传递参数，可以修改train.py添加命令行参数支持。以下是修改方法：

1. 在train.py中添加argparse支持：

```python
import argparse

# 在文件末尾修改
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Industrial Defect Detection Training')
    parser.add_argument('--arch', default='resnet50', help='网络架构')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epoch', type=int, default=50, help='训练轮数')
    parser.add_argument('--data-root', default='dataset', help='数据集根路径')
    parser.add_argument('--train-list', default='dataset/train_small.txt', help='训练集列表')
    parser.add_argument('--val-list', default='dataset/test_small.txt', help='验证集列表')
    parser.add_argument('--gpus', default='0', help='GPU设备ID')
    parser.add_argument('--num-classes', type=int, default=9, help='类别数')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行训练
    main(args)
```

2. 然后可以这样运行：

```bash
python train.py --arch resnet50 --batch-size 16 --lr 0.0001 --epoch 100
```

## 4. 监控训练过程

训练过程中会输出以下信息：
- 训练轮数（Epoch）
- 迭代次数（Iteration）
- 训练损失（Loss）
- 学习率（LR）
- 验证准确率（Val Acc）

示例输出：

```
Use GPU: 0 for training.
Epoch: 0, Iter: 0, Loss: 2.1972, LR: 0.0010
Epoch: 0, Iter: 5, Loss: 2.1845, LR: 0.0010
...
val...
Val Loss: 1.8752, Val Acc: 0.3500
```

## 5. 查看训练结果

训练完成后，模型权重会保存在`checkpoint_mdf_rgb`目录下：
- `checkpoint_epoch_xxxx.pth.tar`：每轮训练的模型权重
- `checkpoint_best.path.tar`：验证准确率最高的模型权重

## 6. 常见问题解决

### 6.1 GPU显存不足
- 减小batch_size参数
- 使用更小的网络架构（如resnet18代替resnet50）

### 6.2 训练速度慢
- 增加num_workers参数（数据加载线程数）
- 使用更大的batch_size（如果显存允许）

### 6.3 准确率低
- 增加训练轮数（epoch）
- 调整学习率策略
- 尝试数据增强

## 7. 高级选项

### 7.1 数据增强
可以在`dataset.py`中修改数据加载器，添加数据增强操作：

```python
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 7.2 学习率调整
可以在`train.py`中修改学习率调整策略：

```python
# 使用余弦退火学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
```

---

现在您已经了解了如何手动运行训练模型。根据您的具体需求，可以调整相应的参数和配置来获得更好的训练效果。