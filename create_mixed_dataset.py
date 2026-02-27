import cv2
import os
import shutil
from tqdm import tqdm
from image_utils import ImagePreprocessor

# ================= 配置区域 =================
BASE_DIR = r'D:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\NEU-DET'
SRC_TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'train', 'images')
SRC_TRAIN_LABEL_DIR = os.path.join(BASE_DIR, 'train', 'labels')
SRC_VAL_IMG_DIR = os.path.join(BASE_DIR, 'validation', 'images')
SRC_VAL_LABEL_DIR = os.path.join(BASE_DIR, 'validation', 'labels')

# 混合数据集目录 (Mixed = Original + Enhanced)
DST_BASE_DIR = r'D:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\NEU-DET_Mixed'
DST_TRAIN_IMG_DIR = os.path.join(DST_BASE_DIR, 'train', 'images')
DST_TRAIN_LABEL_DIR = os.path.join(DST_BASE_DIR, 'train', 'labels')
DST_VAL_IMG_DIR = os.path.join(DST_BASE_DIR, 'validation', 'images')
DST_VAL_LABEL_DIR = os.path.join(DST_BASE_DIR, 'validation', 'labels')

def process_dataset_split(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, split_name):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    if not os.path.exists(src_img_dir):
        print(f"⚠️ 警告: 源目录不存在 {src_img_dir}")
        return

    print(f"🔍 正在扫描目录: {src_img_dir}")
    # 递归查找所有图片文件 (支持子文件夹结构)
    img_files_with_relpath = []
    for root, dirs, files in os.walk(src_img_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.bmp')):
                # 获取相对于 src_img_dir 的路径 (例如: 'crazing/crazing_1.jpg')
                rel_path = os.path.relpath(os.path.join(root, file), src_img_dir)
                img_files_with_relpath.append(rel_path)

    print(f"\n🚀 开始构建混合 {split_name} 集，预计共 {len(img_files_with_relpath) * 2} 张图片 (原图 + 增强图)...")
    
    # 初始化预处理器
    preprocessor = ImagePreprocessor()

    for rel_path in tqdm(img_files_with_relpath):
        src_img_path = os.path.join(src_img_dir, rel_path)
        img = cv2.imread(src_img_path)
        
        if img is None:
            continue
            
        # 1. 保存原图 (Origin)
        # 为了避免文件名冲突（虽然原图和增强图文件名通常一样，但在混合目录里需要区分），
        # 我们给原图加个前缀或者后缀，或者给增强图加后缀。
        # 这里选择：原图保持原名，增强图加 _aug 后缀 (或者相反，为了方便后续管理，建议增强图改名)
        # 但是 YOLO 读取标签是找同名的 txt。
        # 所以：
        # - 原图: image.jpg -> 对应 label.txt
        # - 增强图: image_aug.jpg -> 对应 label_aug.txt (复制一份 label 改名)
        
        dst_img_path_origin = os.path.join(dst_img_dir, rel_path)
        os.makedirs(os.path.dirname(dst_img_path_origin), exist_ok=True)
        # 直接复制原图，比 imwrite 更快且不损失
        shutil.copy(src_img_path, dst_img_path_origin)

        # 2. 生成增强图 (Enhanced)
        processed_img = preprocessor.process(img)
        
        # 构造增强图的文件名: name.jpg -> name_aug.jpg
        file_name, file_ext = os.path.splitext(rel_path)
        aug_rel_path = f"{file_name}_aug{file_ext}"
        dst_img_path_aug = os.path.join(dst_img_dir, aug_rel_path)
        
        cv2.imwrite(dst_img_path_aug, processed_img)

        # 3. 处理标签
        # 找到原始标签
        label_rel_path = os.path.splitext(rel_path)[0] + '.txt'
        src_label_path = os.path.join(src_label_dir, label_rel_path)
        
        # 兼容性查找标签
        if not os.path.exists(src_label_path):
             flat_label_name = os.path.basename(label_rel_path)
             src_label_path_flat = os.path.join(src_label_dir, flat_label_name)
             if os.path.exists(src_label_path_flat):
                 src_label_path = src_label_path_flat

        if os.path.exists(src_label_path):
            # 3.1 复制给原图的标签
            dst_label_path_origin = os.path.join(dst_label_dir, label_rel_path)
            os.makedirs(os.path.dirname(dst_label_path_origin), exist_ok=True)
            shutil.copy(src_label_path, dst_label_path_origin)
            
            # 3.2 复制给增强图的标签 (文件名要对应: name_aug.txt)
            label_name, label_ext = os.path.splitext(label_rel_path)
            aug_label_rel_path = f"{label_name}_aug{label_ext}"
            dst_label_path_aug = os.path.join(dst_label_dir, aug_label_rel_path)
            shutil.copy(src_label_path, dst_label_path_aug)

def main():
    print("开始构建混合数据集 (Original + Enhanced)...")
    process_dataset_split(SRC_TRAIN_IMG_DIR, SRC_TRAIN_LABEL_DIR, 
                         DST_TRAIN_IMG_DIR, DST_TRAIN_LABEL_DIR, "Train")
    process_dataset_split(SRC_VAL_IMG_DIR, SRC_VAL_LABEL_DIR, 
                         DST_VAL_IMG_DIR, DST_VAL_LABEL_DIR, "Validation")
    print("\n✅ 混合数据集构建完成！输出目录: NEU-DET_Mixed")

if __name__ == '__main__':
    main()
