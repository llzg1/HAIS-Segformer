import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ==============================================================================
# 经典 SOTA 模型工厂 (Model Factory)
# ==============================================================================
# 这里的 encoder_name='resnet50' 是为了保证和你的 SegFormer (B0/B1) 在参数量上尽量公平
# 如果你的 B0 很小，也可以把这里改成 'resnet18' 或 'resnet34'

def get_sota_model(model_name, num_classes=2, in_channels=3, encoder='resnet50'):
    print(f"Creating {model_name} with {encoder} backbone...")

    if model_name.lower() == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes
        )

    elif model_name.lower() == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes
        )

    elif model_name.lower() == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes
        )

    elif model_name.lower() == 'fcn':
        # FCN 通常用 ResNet 不带空洞卷积，这里用 U-Net 库里的 FCN 变体
        # 或者直接使用 torchvision 的 fcn_resnet50
        from torchvision import models
        # 注意：torchvision 的 FCN 稍微难改通道数，建议直接用 smp 的实现
        model = smp.FPN(  # FPN 是 FCN 的进阶版，效果更好，也可以代表经典卷积网络
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes
        )

    elif model_name.lower() == 'segnet':
        # SMP 没有直接的 SegNet (太老了)，可以用 UNet++ 代替作为强基线
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


if __name__ == "__main__":
    # 1. 创建模型
    net = get_sota_model('deeplabv3+', num_classes=2, encoder='resnet18')

    # 2. 【关键修改】切换到评估模式 (Evaluation Mode)
    # 这会告诉 BatchNorm 层使用预训练好的统计值，而不是从当前输入计算
    net.eval()

    # 3. 创建假数据 (BatchSize=1)
    dummy = torch.randn(1, 3, 512, 512)

    # 4. 前向传播 (建议加上 no_grad 以节省显存)
    with torch.no_grad():
        out = net(dummy)

    print(f"Output shape: {out.shape}")