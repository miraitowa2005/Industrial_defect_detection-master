# coding: utf-8

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import time
import os
import shutil
import math

import models
from config import Args
from dataset import ImageListDataset


best_acc = 0

# 定义Focal Loss，用于解决类别不平衡问题
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_learning_rate(optimizer, epoch, iter_size, iter_num, args):
    current_iter = epoch * iter_size + iter_num
    if current_iter < args.warm_up:
        # 使用更平缓的warm up曲线
        current_lr = args.lr * math.pow(current_iter / args.warm_up, 2)
    else:
        # 改进的余弦退火策略，增加学习率衰减的速度
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - args.warm_up / iter_size) / (args.epoch - args.warm_up / iter_size)))
        current_lr = args.lr * cosine_decay * 0.1 + 0.00001  # 添加最小学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


from tqdm import tqdm

def train(dataloader, model, criterion, optimizer, epoch, args, scaler=None):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    tic = time.time()
    # 使用tqdm添加进度条
    for i, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} - Training", unit="batch")):
        lr = set_learning_rate(optimizer, epoch, len(dataloader), i, args)
        batch_size = images.size(0)

        # 使用模型所在的设备
        device = next(model.parameters()).device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if scaler is not None:
            with autocast():
                outputs = model(images) # shape=(b, n_classes)
                loss = criterion(outputs, labels)
            losses.update(loss.item(), batch_size)
            
            # 使用scaler进行反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准精度训练
            outputs = model(images) # shape=(b, n_classes)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), batch_size)
            
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            # 获取GPU内存使用情况
            gpu_memory_allocated = 0
            gpu_memory_cached = 0
            if device.type == 'cuda':
                gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024**3
                
            print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
            print('Train Epoch: [{0}][{1}/{2}] '
                  'Batch Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Loss {loss.val:.3f}({loss.avg:.3f}) '
                  'Lr {lr:.6f} '
                  'GPU Mem: {gpu_mem:.2f} GB / {gpu_cache:.2f} GB'
                  .format(epoch, i, len(dataloader),
                          batch_time=batch_time,
                          loss=losses,
                          lr=lr,
                          gpu_mem=gpu_memory_allocated,
                          gpu_cache=gpu_memory_cached), flush=True)


def val(dataloader, model, criterion, args, scaler=None):
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.eval()

    with torch.no_grad():
        # 使用tqdm添加进度条
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Validation", unit="batch")):
            batch_size = images.size(0)
            # 使用模型所在的设备
            device = next(model.parameters()).device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 混合精度验证
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            losses.update(loss.item(), batch_size)

            _, preds = torch.max(outputs, 1)
            acc = torch.mean((preds == labels.data).float())

            accuracy.update(acc.item(), batch_size)

            if i % args.print_freq == 0:
                print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
                print('Val: [{0}/{1}] '
                      'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                      'Acc: {acc.val:.3f}({acc.avg:.3f})'
                      .format(i, len(dataloader),
                              loss=losses,
                              acc=accuracy), flush=True)

    return losses.avg, accuracy.avg


def main(args):
    # 检查是否有可用的CUDA设备
    use_cuda = torch.cuda.is_available() and args.gpus
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print("Use GPU: {} for training.".format(args.gpus))
        print("GPU名称: {}".format(torch.cuda.get_device_name(0)))
        print("GPU内存总量: {:.2f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
    else:
        print("Use CPU for training.")

    # model - 手动加载预训练权重
    import torchvision.models as torch_models
    
    # 对于我们自己实现的模型，使用models.__dict__
    if args.arch in models.__dict__:
        model = models.__dict__[args.arch](pretrained=False)
    # 对于torchvision中没有重写的模型，直接使用torch_models
    elif args.arch in torch_models.__dict__:
        model = torch_models.__dict__[args.arch](pretrained=False)
    else:
        raise ValueError(f"不支持的模型架构: {args.arch}")
    
    # 根据不同模型结构获取特征提取层和分类层
    if hasattr(model, 'fc'):  # ResNet, ResNeXt等
        num_ftrs = model.fc.in_features
        # 添加Dropout层作为正则化
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, args.num_classes)
        )
        classifier_name = 'fc'
    elif hasattr(model, 'classifier'):  # EfficientNet, MobileNetV3等
        # 处理EfficientNet的分类器结构
        if args.arch.startswith('efficientnet'):
            num_ftrs = model.classifier[1].in_features
            # 在分类器中添加Dropout
            model.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, args.num_classes)
            )
        # 处理MobileNetV3的分类器结构
        elif args.arch.startswith('mobilenet_v3'):
            num_ftrs = model.classifier[3].in_features
            # 在分类器中添加Dropout
            model.classifier[3] = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, args.num_classes)
            )
        classifier_name = 'classifier'
    else:
        raise ValueError(f"模型 {args.arch} 没有标准的分类层结构")
    
    # 手动加载预训练权重并处理差异
    if args.arch in torch_models.__dict__:
        # 加载原始模型的预训练权重
        pretrained_model = torch_models.__dict__[args.arch](pretrained=True)
        pretrained_state_dict = pretrained_model.state_dict()
        
        # 复制预训练权重，跳过最后的分类层
        model_state_dict = model.state_dict()
        for name, param in pretrained_state_dict.items():
            # 跳过分类层的权重
            if classifier_name == 'fc' and (name == 'fc.weight' or name == 'fc.bias'):
                continue
            elif classifier_name == 'classifier':
                if args.arch.startswith('efficientnet') and name.startswith('classifier.1'):
                    continue
                elif args.arch.startswith('mobilenet_v3') and name.startswith('classifier.3'):
                    continue
            
            if name in model_state_dict:
                try:
                    model_state_dict[name].copy_(param)
                except:
                    print(f"跳过权重: {name} (形状不匹配)")
    
    print("预训练权重加载完成")

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        # 回退到标准的交叉熵损失，比Focal Loss更稳定
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        model = model.cpu()
        criterion = nn.CrossEntropyLoss().cpu()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    
    # 初始化混合精度训练
    scaler = GradScaler() if use_cuda else None

    if args.checkpoint:
        print('=> loading checkpoint from {}...'.format(args.checkpoint))
        state = torch.load(args.checkpoint)
        args.start_epoch = state['epoch']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    # 优化数据增强策略：去除模糊，减少过度裁剪
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        # 改为RandomCrop以保留原始比例，避免缺陷变形
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # 减弱颜色抖动，工业环境光照通常受控
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        # 使用适合RGB图像的归一化参数
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = ImageListDataset(args.data_root, args.train_list, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    if args.val_list:
        # val dataset
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # 使用适合RGB图像的归一化参数
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_dataset = ImageListDataset(args.data_root, args.val_list, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    # 早停策略参数
    patience = 10
    no_improve_epoch = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.start_epoch, args.epoch):
        global best_acc
        train(train_loader, model, criterion, optimizer, epoch, args, scaler)
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(args.checkpoint_dir,
                                       'checkpoint_epoch_{:04d}.pth.tar'.format(state['epoch']))
        torch.save(state, checkpoint_file)

        if args.val_list:
            print('val...')
            val_loss, val_acc = val(val_loader, model, criterion, args, scaler)
            print('Val Loss: {loss:.3f}, Val Acc: {acc:.3f}'.format(loss=val_loss, acc=val_acc))

            is_best = (val_acc > best_acc)
            if is_best:
                best_acc = val_acc
                best_checkpoint_file = os.path.join(args.checkpoint_dir,
                                                    'checkpoint_best.pth.tar')
                shutil.copy2(checkpoint_file, best_checkpoint_file)
                print(f'New best accuracy: {best_acc:.3f}, checkpoint saved.')
                no_improve_epoch = 0  # 重置早停计数器
            else:
                no_improve_epoch += 1
                print(f'No improvement for {no_improve_epoch} epochs.')
            
            # 早停检查
            if no_improve_epoch >= patience:
                print(f'Early stopping after {epoch+1} epochs.')
                break


if __name__ == '__main__':
    main(Args)
