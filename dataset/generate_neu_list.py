import os
import glob

def generate_list(root_dir, output_file, class_mapping):
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历所有类别文件夹
        for class_name, class_id in class_mapping.items():
            # 查找该类别下的所有图片
            # 假设结构是 root_dir/images/class_name/*.jpg (或.bmp, .png)
            # 注意：用户提供的结构是 NEU-DET/train/images/class_name
            # 所以这里传入的 root_dir 应该是 .../NEU-DET/train 或 .../NEU-DET/validation
            
            image_dir = os.path.join(root_dir, 'images', class_name)
            if not os.path.exists(image_dir):
                print(f"Warning: Directory not found: {image_dir}")
                continue
                
            # 支持常见图片格式
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
            images = []
            for ext in extensions:
                images.extend(glob.glob(os.path.join(image_dir, ext)))
            
            print(f"Found {len(images)} images for class {class_name} in {root_dir}")
            
            for img_path in images:
                # 获取相对于 NEU-DET 根目录的路径
                # 假设 data_root 设置为 NEU-DET 文件夹
                # 那么 txt 中的路径应该是 train/images/class_name/image.jpg
                
                # 获取路径的最后几部分: split_folder/images/class_name/filename
                # 例如: train/images/crazing/crazing_1.jpg
                
                # 统一路径分隔符
                img_path = img_path.replace('\\', '/')
                
                # 找到 split_folder (train 或 validation) 的位置
                if '/train/' in img_path:
                    rel_path = img_path.split('/train/')[1]
                    rel_path = 'train/' + rel_path
                elif '/validation/' in img_path:
                    rel_path = img_path.split('/validation/')[1]
                    rel_path = 'validation/' + rel_path
                else:
                    # Fallback just in case
                    rel_path = img_path
                
                f.write(f"{rel_path} {class_id}\n")

def main():
    neu_det_root = r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\NEU-DET'
    dataset_dir = r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset'
    
    # 定义类别映射
    classes = [
        'crazing',
        'inclusion',
        'patches',
        'pitted_surface',
        'rolled-in_scale',
        'scratches'
    ]
    class_mapping = {cls: i for i, cls in enumerate(classes)}
    print(f"Class mapping: {class_mapping}")
    
    # 生成训练集列表
    train_root = os.path.join(neu_det_root, 'train')
    train_list_file = os.path.join(dataset_dir, 'train_neu.txt')
    print(f"Generating {train_list_file}...")
    generate_list(train_root, train_list_file, class_mapping)
    
    # 生成验证集列表
    val_root = os.path.join(neu_det_root, 'validation')
    val_list_file = os.path.join(dataset_dir, 'val_neu.txt')
    print(f"Generating {val_list_file}...")
    generate_list(val_root, val_list_file, class_mapping)
    
    print("Done!")

if __name__ == '__main__':
    main()
