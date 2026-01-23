# coding: utf-8

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import time
import argparse
import numpy as np

import models
from dataset import ImageListDataset
from train import AverageMeter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

cls_list = ['0_scratch',
            '1_gline',
            '2_bubble',
            '3_defect',
            '4_unformed',
            '5_foreign_matter',
            '6_burr',
            '7_lr',
            '8_pin']

def parse():
    args = argparse.ArgumentParser('model eval')
    args.add_argument('dataroot', type=str,
                    help='testset root dir')
    args.add_argument('testlist', type=str,
                    help='testset list file')
    args.add_argument('checkpoint', type=str,
                    help='checkpoint path')
    args.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    args.add_argument('--batch_size', type=int, default=64,
                    help='batch_size, default=64')
    args.add_argument('--num_workers', type=int, default=4,
                    help='DataLoader readers. default=4')
    args.add_argument('--include_classes', nargs='*', type=int, default=None,
                    help='只考虑这些类别的准确率计算，例如 --include_classes 0 1 2 3 4 5')

    return args.parse_args()


def cal_acc(dataloader, model, num_classes, device, include_classes=None):
    accuracy = AverageMeter()
    model.eval()

    cls_count = np.zeros(num_classes, dtype=np.float32)
    cls_correct = np.zeros(num_classes, dtype=np.float32)

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            for gt_label in labels:
                cls_count[int(gt_label.item())] += 1

            _, preds = torch.max(outputs, 1)
            for corr_pred in labels[preds == labels.data]:
                cls_correct[int(corr_pred.item())] += 1

            acc = torch.mean((preds == labels.data).float())
            cls_acc = cls_correct / (cls_count + 1e-8)

            accuracy.update(acc.item(), batch_size)

            print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
            print('Test: [{0}/{1}] '
                  'Acc: {acc.val:.3f}({acc.avg:.3f})'
                  .format(i + 1, len(dataloader),
                          acc=accuracy), flush=True)

    # 计算指定类别的整体准确率
    if include_classes is None:
        # 如果没有指定，只考虑有样本的类别
        include_classes = np.where(cls_count > 0)[0]
    else:
        # 确保只包含有样本的指定类别
        include_classes = np.array([c for c in include_classes if cls_count[c] > 0])

    if len(include_classes) > 0:
        total_included_samples = np.sum(cls_count[include_classes])
        total_correct_included = np.sum(cls_correct[include_classes])
        included_acc = total_correct_included / total_included_samples
    else:
        included_acc = 0.0

    return included_acc, cls_acc


def eval(data_root, data_list, checkpoint, arch, batch_size, num_workers, include_classes=None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # 使用与训练一致的RGB图像归一化参数
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = ImageListDataset(data_root, data_list, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    # model - 与train.py保持一致的模型加载逻辑
    import torchvision.models as torch_models
    
    # 对于我们自己实现的模型，使用models.__dict__
    if arch in models.__dict__:
        model = models.__dict__[arch](pretrained=False)
    # 对于torchvision中没有重写的模型，直接使用torch_models
    elif arch in torch_models.__dict__:
        model = torch_models.__dict__[arch](pretrained=False)
    else:
        raise ValueError(f"不支持的模型架构: {arch}")
    
    # 根据不同模型结构获取特征提取层和分类层
    if hasattr(model, 'fc'):  # ResNet, ResNeXt等
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(cls_list))
    elif hasattr(model, 'classifier'):  # EfficientNet, MobileNetV3等
        # 处理EfficientNet的分类器结构
        if arch.startswith('efficientnet'):
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, len(cls_list))
        # 处理MobileNetV3的分类器结构
        elif arch.startswith('mobilenet_v3'):
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, len(cls_list))
    else:
        raise ValueError(f"模型 {arch} 没有标准的分类层结构")
    
    model = model.to(device)

    state = torch.load(checkpoint, map_location=device)
    state_dict = dict()
    for k, v in state['model'].items():
        state_dict[k.replace('module.','')] = v
    model.load_state_dict(state_dict)

    acc, cls_acc, precision, recall, f1 = cal_acc(test_loader, model, len(cls_list), device, include_classes)

    print('=> Test Acc: %.4f\n' % acc)
    print(f"{'Class':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    for i in range(len(cls_list)):
        print(f"{cls_list[i]:<20} {cls_acc[i]:.4f}     {precision[i]:.4f}     {recall[i]:.4f}     {f1[i]:.4f}")


if __name__ == '__main__':
    args = parse()
    print('Eval...')
    eval(args.dataroot, args.testlist, args.checkpoint, args.arch, args.batch_size, args.num_workers, args.include_classes)
