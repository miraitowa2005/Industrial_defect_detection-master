import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import glob
from PIL import Image


class ImageListDataset(Dataset):
    def __init__(self, data_root, data_list, transform=None):
        super(ImageListDataset, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.img_list = []

        # 尝试使用不同的编码格式打开文件
        encodings = ['utf-8', 'gbk', 'gb2312']
        for encoding in encodings:
            try:
                with open(data_list, 'r', encoding=encoding) as f:
                    self.img_list = [line.strip() for line in f]
                print(f'成功使用{encoding}编码打开文件')
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError(f'无法使用以下编码打开文件: {encodings}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # 正确解析数据列表格式：图像路径 类别ID
        line = self.img_list[index].strip()
        img_path, label_str = line.split(' ', 1)  # 按第一个空格分割
        label = int(label_str)
        img_fn = os.path.join(self.data_root, img_path)

        img = Image.open(img_fn)
        # 将所有图像转换为RGB格式（三通道）以充分利用预训练模型
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == '__main__':
    dataset = ImageListDataset('/workspace/personal/classification/dataset',
                               '/workspace/personal/classification/dataset/train.txt')
    img, label = dataset[10]
