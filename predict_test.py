from ultralytics import YOLO
import torch

# PyTorch 2.6+ 兼容性修复：添加安全白名单
try:
    from torch.serialization import add_safe_globals
    import torch.nn as nn
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn import modules as um
    add_safe_globals([
        DetectionModel,
        nn.Sequential,
        getattr(um, "Conv", None),
        getattr(um, "RepConv", None),
        getattr(um, "C2f", None),
        getattr(um, "Bottleneck", None),
        getattr(um, "SPPF", None),
    ])
except Exception:
    pass

model = YOLO("runs/detect/train6/weights/best.pt")
source = "NEU-DET/validation/images/crazing/crazing_255.jpg"
results = model(source, save=True, show=True)
print("预测完成，模型可用！")
