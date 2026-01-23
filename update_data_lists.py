import os
import random

# 数据集根目录
DATA_ROOT = 'd:\\MachineLearning\\ComputerVersion\\Industrial_defect_detection-master\\dataset'

# 类别列表
CLASSES = [
    '0_scratch',
    '1_gline',
    '2_bubble',
    '3_defect',
    '4_unformed',
    '5_foreign_matter',
    '6_burr',
    '7_lf',
    '8_pin'
]

# 生成的图像数量
NUM_IMAGES_PER_CLASS = 100  # 这是我们实际生成的数量

# 生成新的训练和测试列表
all_images = []

for class_name in CLASSES:
    class_dir = os.path.join(DATA_ROOT, class_name)
    
    # 获取该类别的所有图像文件
    images = []
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg'):
            images.append(f'{class_name}/{filename}')
    
    # 确保我们只使用前NUM_IMAGES_PER_CLASS个图像
    images = images[:NUM_IMAGES_PER_CLASS]
    all_images.extend(images)

# 随机打乱图像列表
random.shuffle(all_images)

# 将80%的数据用于训练，20%用于测试
train_size = int(0.8 * len(all_images))
train_images = all_images[:train_size]
test_images = all_images[train_size:]

# 更新train_small.txt文件
train_file_path = os.path.join(DATA_ROOT, 'train_small.txt')
with open(train_file_path, 'w', encoding='utf-8') as f:
    for img_path in train_images:
        f.write(f'{img_path}\n')

# 更新test_small.txt文件
test_file_path = os.path.join(DATA_ROOT, 'test_small.txt')
with open(test_file_path, 'w', encoding='utf-8') as f:
    for img_path in test_images:
        f.write(f'{img_path}\n')

print(f'Generated {len(train_images)} training images and {len(test_images)} test images.')
print(f'Updated train_small.txt and test_small.txt files.')
print(f'Example train images: {train_images[:5]}')
print(f'Example test images: {test_images[:5]}')
print("\n=== 数据列表更新完成 ===")