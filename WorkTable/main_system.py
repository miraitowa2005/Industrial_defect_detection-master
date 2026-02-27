import sys
import cv2
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QFrame)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
from qt_material import apply_stylesheet

# --- 核心配置 ---
MODEL_PATH = 'best.pt'  # 你的模型路径
# 对应你的 neu_det.yaml 里的类别
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

class DetectionThread(QThread):
    change_pixmap_signal = Signal(QImage)
    update_result_signal = Signal(dict, bool) # 传递检测结果计数和是否有缺陷

    def __init__(self):
        super().__init__()
        self.running = False
        self.mode = 'camera' # camera, video, image
        self.file_path = ''
        self.model = YOLO(MODEL_PATH)
        self.cap = None

    def run(self):
        self.running = True
        
        # 模式初始化
        if self.mode == 'camera':
            self.cap = cv2.VideoCapture(0) # 0 代表默认摄像头，工业相机通常是 0 或 1
        elif self.mode == 'video':
            self.cap = cv2.VideoCapture(self.file_path)
        elif self.mode == 'image':
            self.process_image()
            return

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.mode == 'video': # 视频播完循环播放
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            self.inference_and_emit(frame)

        if self.cap:
            self.cap.release()

    def process_image(self):
        frame = cv2.imread(self.file_path)
        if frame is not None:
            self.inference_and_emit(frame)

    def inference_and_emit(self, frame):
        # 1. YOLO 推理
        results = self.model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        res_plotted = results[0].plot()
        
        # 2. 统计结果
        cls_count = {}
        has_defect = False
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            cls_count[name] = cls_count.get(name, 0) + 1
            has_defect = True
        
        self.update_result_signal.emit(cls_count, has_defect)

        # 3. 转换图像格式以在 Qt 中显示
        rgb_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放以适应 UI
        p = convert_to_qt_format.scaled(1280, 720, Qt.KeepAspectRatio)
        self.change_pixmap_signal.emit(p)

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEU-DET 工业缺陷智能检测系统 | Industrial Defect Detection System")
        self.resize(1400, 850)

        # 布局容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左侧：显示区域 (80%) ---
        left_layout = QVBoxLayout()
        
        # 1. 标题栏
        self.title_label = QLabel("实时监控画面 / Monitor View")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.title_label)

        # 2. 视频/图片显示框
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #3d3d3d; background-color: #1e1e1e; border-radius: 10px;")
        self.video_label.setMinimumSize(960, 540)
        left_layout.addWidget(self.video_label)

        # 3. 状态栏 (报警指示)
        self.status_bar = QLabel("系统就绪 / System Ready")
        self.status_bar.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("background-color: #2c3e50; color: white; padding: 10px; border-radius: 5px;")
        left_layout.addWidget(self.status_bar)

        main_layout.addLayout(left_layout, 7) # 占比 7

        # --- 右侧：控制与数据区 (30%) ---
        right_layout = QVBoxLayout()
        
        # 1. 结果统计表
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["缺陷类型", "数量"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(QLabel("检测结果统计 / Statistics"))
        right_layout.addWidget(self.result_table)

        # 2. 功能按钮区
        btn_layout = QVBoxLayout()
        
        self.btn_camera = QPushButton("🎥 连接工业摄像头")
        self.btn_camera.setMinimumHeight(50)
        self.btn_camera.clicked.connect(self.start_camera)
        
        self.btn_video = QPushButton("📼 上传视频文件")
        self.btn_video.setMinimumHeight(50)
        self.btn_video.clicked.connect(self.upload_video)

        self.btn_image = QPushButton("🖼️ 上传图片检测")
        self.btn_image.setMinimumHeight(50)
        self.btn_image.clicked.connect(self.upload_image)

        self.btn_stop = QPushButton("🛑 停止检测")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white;")
        self.btn_stop.clicked.connect(self.stop_detection)

        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_image)
        btn_layout.addWidget(self.btn_stop)
        
        right_layout.addLayout(btn_layout)
        main_layout.addLayout(right_layout, 3) # 占比 3

        # 逻辑线程
        self.thread = DetectionThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_result_signal.connect(self.update_table)

    def start_camera(self):
        self.stop_detection()
        self.thread.mode = 'camera'
        self.title_label.setText("工业相机实时采集 / Live Camera Feed")
        self.thread.start()

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.stop_detection()
            self.thread.mode = 'video'
            self.thread.file_path = file_path
            self.title_label.setText(f"视频回放：{file_path.split('/')[-1]}")
            self.thread.start()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.png *.bmp)")
        if file_path:
            self.stop_detection()
            self.thread.mode = 'image'
            self.thread.file_path = file_path
            self.title_label.setText(f"图片检测：{file_path.split('/')[-1]}")
            self.thread.start()

    def stop_detection(self):
        if self.thread.isRunning():
            self.thread.stop()
        self.status_bar.setText("检测已停止 / Stopped")
        self.status_bar.setStyleSheet("background-color: #2c3e50; color: white; padding: 10px;")

    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_table(self, result_dict, has_defect):
        # 更新表格
        self.result_table.setRowCount(0)
        for name, count in result_dict.items():
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(name))
            self.result_table.setItem(row, 1, QTableWidgetItem(str(count)))
        
        # 报警逻辑 (任务书要求)
        if has_defect:
            self.status_bar.setText(f"⚠️ 警告：检测到 {sum(result_dict.values())} 处缺陷！")
            self.status_bar.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px; font-weight: bold;")
        else:
            self.status_bar.setText("✅ 检测正常 / OK")
            self.status_bar.setStyleSheet("background-color: #27ae60; color: white; padding: 10px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # --- 核心美化：应用暗黑医疗/工业主题 ---
    # 可选主题：'dark_teal.xml', 'dark_cyan.xml', 'dark_medical.xml'
    apply_stylesheet(app, theme='dark_medical.xml')
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())