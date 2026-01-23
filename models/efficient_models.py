import torch.nn as nn
import torch
import torchvision.models as torch_models

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
           'mobilenet_v3_small', 'mobilenet_v3_large', 'resnext50_32x4d', 'resnext101_32x8d']


def efficientnet_b0(pretrained=False, **kwargs):
    """Constructs an EfficientNet B0 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torch_models.efficientnet_b0(pretrained=pretrained, **kwargs)
    return model


def efficientnet_b1(pretrained=False, **kwargs):
    """Constructs an EfficientNet B1 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torch_models.efficientnet_b1(pretrained=pretrained, **kwargs)
    return model


def efficientnet_b2(pretrained=False, **kwargs):
    """Constructs an EfficientNet B2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torch_models.efficientnet_b2(pretrained=pretrained, **kwargs)
    return model


def mobilenet_v3_small(pretrained=False, **kwargs):
    """Constructs a MobileNetV3 Small model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torch_models.mobilenet_v3_small(pretrained=pretrained, **kwargs)
    return model


def mobilenet_v3_large(pretrained=False, **kwargs):
    """Constructs a MobileNetV3 Large model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torch_models.mobilenet_v3_large(pretrained=pretrained, **kwargs)
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torch_models.resnext50_32x4d(pretrained=pretrained, **kwargs)
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torch_models.resnext101_32x8d(pretrained=pretrained, **kwargs)
    return model
