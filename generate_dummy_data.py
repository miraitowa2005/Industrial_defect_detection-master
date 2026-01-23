import os
import numpy as np
from PIL import Image

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

# 生成虚拟图像
for class_name in CLASSES:
    class_dir = os.path.join(DATA_ROOT, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # 为每个类别生成少量虚拟图像
    for i in range(50):
        # 生成224x224的随机图像
        img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # 保存图像
        if class_name == '6_burr':
            img_name = f'BURR毛边{i}.jpg'
        elif class_name == '2_bubble':
            img_name = f'qipao{i}.jpg'
        elif class_name == '7_lf':
            img_name = f'LF{i}.jpg'
        elif class_name == '8_pin':
            img_name = f'pin{i}.jpg'
        else:
            img_name = f'{class_name.split("_", 1)[1]}{i}.jpg'
        
        img_path = os.path.join(class_dir, img_name)
        img.save(img_path)
    
    print(f'Generated 50 dummy images for class: {class_name}')

print('\nDummy dataset generated successfully!')