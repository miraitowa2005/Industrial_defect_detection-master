import cv2
import os
import numpy as np
import shutil
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始数据集路径
BASE_DIR = r'D:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\NEU-DET'
SRC_TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'train', 'images')
SRC_TRAIN_LABEL_DIR = os.path.join(BASE_DIR, 'train', 'labels')
SRC_VAL_IMG_DIR = os.path.join(BASE_DIR, 'validation', 'images')
SRC_VAL_LABEL_DIR = os.path.join(BASE_DIR, 'validation', 'labels')

# 2. 输出的新数据集路径
DST_BASE_DIR = r'D:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\NEU-DET_Enhanced'
DST_TRAIN_IMG_DIR = os.path.join(DST_BASE_DIR, 'train', 'images')
DST_TRAIN_LABEL_DIR = os.path.join(DST_BASE_DIR, 'train', 'labels')
DST_VAL_IMG_DIR = os.path.join(DST_BASE_DIR, 'validation', 'images')
DST_VAL_LABEL_DIR = os.path.join(DST_BASE_DIR, 'validation', 'labels')

# ================= 图像增强核心函数 =================
def process_image(img):
    """
    针对 Crazing 和 Scratches 的专项增强组合拳
    """
    # 1. 转灰度 (虽然 YOLO 也可以吃 RGB，但工业缺陷看灰度就够了，且处理更快)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. 高斯降噪 (去除传感器噪点，避免 CLAHE 放大噪声)
    # kernel=(3,3) 是为了保留细微裂纹，不要用 (5,5) 或更大，否则裂纹会糊掉
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. CLAHE 局部直方图均衡化 (核心步骤！)
    # clipLimit=2.5: 对比度限制。数值越大对比越强烈，2.0-3.0 适合钢铁表面。
    # tileGridSize=(8,8): 局部窗口大小。
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. 边缘锐化 (Sharpening) - 专门针对 Scratches
    # 使用拉普拉斯算子或自定义锐化核
    kernel_sharpen = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])
    sharp = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # 5.以此为基础转回 3 通道 (YOLO 默认输入是 3 通道)
    img_final = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    
    return img_final

def process_dataset_split(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, split_name):
    # 创建输出目录
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    # 获取所有图片文件
    if not os.path.exists(src_img_dir):
        print(f"⚠️ 警告: 源目录不存在 {src_img_dir}")
        return

    img_files = [f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.png', '.bmp'))]
    
    print(f"\n🚀 开始处理 {split_name} 集，共 {len(img_files)} 张图片...")
    print(f"📂 源目录: {src_img_dir}")
    print(f"📂 目标目录: {dst_img_dir}")

    for img_name in tqdm(img_files):
        # --- 1. 处理图片 ---
        src_img_path = os.path.join(src_img_dir, img_name)
        img = cv2.imread(src_img_path)
        
        if img is None:
            continue
            
        # 执行增强
        processed_img = process_image(img)
        
        # 保存增强后的图片
        dst_img_path = os.path.join(dst_img_dir, img_name)
        cv2.imwrite(dst_img_path, processed_img)

        # --- 2. 复制对应的标签文件 ---
        # 假设标签文件名与图片名一致 (只是后缀不同)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label_path = os.path.join(src_label_dir, label_name)
        dst_label_path = os.path.join(dst_label_dir, label_name)

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            # 有些背景图可能没有标签，这很正常
            pass

# ================= 主执行流程 =================
def main():
    print("开始构建增强数据集...")
    
    # 处理训练集
    process_dataset_split(SRC_TRAIN_IMG_DIR, SRC_TRAIN_LABEL_DIR, 
                         DST_TRAIN_IMG_DIR, DST_TRAIN_LABEL_DIR, "Train")
    
    # 处理验证集
    process_dataset_split(SRC_VAL_IMG_DIR, SRC_VAL_LABEL_DIR, 
                         DST_VAL_IMG_DIR, DST_VAL_LABEL_DIR, "Validation")

    print("\n✅ 数据增强完成！")
    print(f"新数据集位置: {DST_BASE_DIR}")
    print("下一步：请使用 neu_det_enhanced.yaml 配置文件进行训练。")

if __name__ == '__main__':
    main()
