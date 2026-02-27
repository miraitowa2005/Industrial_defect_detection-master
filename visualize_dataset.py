import cv2
import os
import glob
from tqdm import tqdm
import yaml

# ================= 配置区域 =================
# 原始数据集根目录
SRC_ROOT = r'D:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\NEU-DET'
# 可视化结果保存目录
DST_ROOT = r'D:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\NEU-DET_Visualized'

# 类别名称 (对应 neu_det.yaml)
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# 类别颜色 (BGR格式)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255)   # Yellow
]

def yolo_to_pixel(x_center, y_center, w, h, img_w, img_h):
    """将 YOLO 归一化坐标转换为像素坐标 (x_min, y_min, x_max, y_max)"""
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    
    x_min = int(x_center - w / 2)
    y_min = int(y_center - h / 2)
    x_max = int(x_center + w / 2)
    y_max = int(y_center + h / 2)
    
    return x_min, y_min, x_max, y_max

def process_dataset():
    print(f"开始生成可视化数据集...")
    print(f"源目录: {SRC_ROOT}")
    print(f"目标目录: {DST_ROOT}")

    # 递归查找所有图片
    img_extensions = ['*.jpg', '*.png', '*.bmp']
    img_files = []
    for ext in img_extensions:
        # 使用 glob 递归查找
        img_files.extend(glob.glob(os.path.join(SRC_ROOT, '**', ext), recursive=True))

    print(f"共找到 {len(img_files)} 张图片")

    for img_path in tqdm(img_files):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]

        # 构造对应的标签路径
        # 假设结构是 images/xxx.jpg -> labels/xxx.txt
        # 或者 images/subdir/xxx.jpg -> labels/subdir/xxx.txt
        # 这里尝试一种通用的替换策略：将路径中的 'images' 替换为 'labels'，后缀换成 .txt
        
        # 获取相对于根目录的路径
        rel_path = os.path.relpath(img_path, SRC_ROOT)
        
        # 尝试推断 label 路径
        # 策略1: 假设标准结构 root/split/images/... -> root/split/labels/...
        parts = rel_path.split(os.sep)
        label_path = None
        
        if 'images' in parts:
            # 替换路径中的 images 为 labels
            label_parts = [p if p != 'images' else 'labels' for p in parts]
            label_rel_path = os.path.join(*label_parts)
            label_rel_path = os.path.splitext(label_rel_path)[0] + '.txt'
            label_path = os.path.join(SRC_ROOT, label_rel_path)
        
        # 如果策略1找不到，尝试策略2: 扁平查找 (在同级目录或特定的labels目录找)
        if not label_path or not os.path.exists(label_path):
            # 简单的同名txt替换
            potential_label = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(potential_label):
                label_path = potential_label
        
        # 绘制标签
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                data = line.strip().split()
                if len(data) >= 5:
                    class_id = int(data[0])
                    x_c, y_c, w, h = map(float, data[1:5])
                    
                    x_min, y_min, x_max, y_max = yolo_to_pixel(x_c, y_c, w, h, img_w, img_h)
                    
                    # 获取颜色和类名
                    color = COLORS[class_id % len(COLORS)]
                    label_text = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
                    
                    # 画框
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # 画标签文字背景
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x_min, y_min - 20), (x_min + text_w, y_min), color, -1)
                    cv2.putText(img, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存图片
        dst_path = os.path.join(DST_ROOT, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img)

    print(f"\n✅ 可视化完成！结果已保存在: {DST_ROOT}")

if __name__ == "__main__":
    process_dataset()
