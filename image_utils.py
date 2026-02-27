import cv2
import numpy as np

class ImagePreprocessor:
    """
    工业视觉专用预处理模块 (统一版)
    针对 NEU-DET 数据集特点设计：去噪 + 纹理增强 + 边缘锐化
    """
    def __init__(self):
        # 1. CLAHE 参数
        # clipLimit: 对比度限制阈值。2.5 适合钢铁表面，能凸显裂纹同时控制噪声。
        self.clahe_clip_limit = 2.5 
        self.clahe_grid_size = (8, 8) 

        # 2. 降噪参数
        # kernel=(3,3) 保留细微裂纹
        self.blur_kernel = (3, 3) 
        self.blur_sigma = 0 

        # 3. 锐化核 (Laplacian 变体)
        self.kernel_sharpen = np.array([[-1, -1, -1], 
                                        [-1,  9, -1], 
                                        [-1, -1, -1]])

    def process(self, image):
        """
        输入: BGR 格式图片
        输出: 预处理后的 BGR 图片 (适配 YOLO)
        """
        if image is None:
            return None

        # 1. 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. 高斯降噪
        denoised = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)

        # 3. CLAHE 增强
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
        enhanced = clahe.apply(denoised)

        # 4. 边缘锐化 (针对划痕和裂纹)
        sharp = cv2.filter2D(enhanced, -1, self.kernel_sharpen)

        # 5. 转回 BGR (YOLO 输入需求)
        processed_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
        
        return processed_bgr
