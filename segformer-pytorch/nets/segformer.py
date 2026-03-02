# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsigmoid = HSigmoid()

    def forward(self, x):
        return x * self.hsigmoid(x)

# -----------------------------------------------------------------------------------
#   CoordAttV2: 坐标注意力 (你的原有模块)
# -----------------------------------------------------------------------------------
class CoordAttV2(nn.Module):
    def __init__(self, inp, reduction=16, norm="gn"):
        super().__init__()
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)

        if norm == "gn":
            g = 8 if mip % 8 == 0 else 4 if mip % 4 == 0 else 1
            self.norm = nn.GroupNorm(g, mip)
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(mip, eps=0.001, momentum=0.03)
        else:
            self.norm = nn.Identity()

        self.act = HSwish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = F.adaptive_avg_pool2d(x, (h, 1))  # [B,C,H,1]
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)  # [B,C,W,1]

        y = torch.cat([x_h, x_w], dim=2)  # [B,C,H+W,1]
        y = self.act(self.norm(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # [B,mip,1,W]

        a_h = torch.sigmoid(self.conv_h(y_h))  # [B,C,H,1]
        a_w = torch.sigmoid(self.conv_w(y_w))  # [B,C,1,W]

        a = a_h * a_w
        return x * (1 + a)  # ✅残差门控


# -----------------------------------------------------------------------------------
#   【新增】CBAM: 通道 + 空间注意力
# -----------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# -----------------------------------------------------------------------------------
#   FeatureInhibition: 水下特征抑制
# -----------------------------------------------------------------------------------
class FeatureInhibition(nn.Module):
    """
    针对水下噪声的特征抑制模块 (Inspired by SMOFFI)
    """

    def __init__(self, channels):
        super().__init__()
        # 使用 1x1 卷积生成抑制门控图
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        # 可学习的抑制系数 alpha
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        noise_gate = self.gate_conv(x)
        x_inhibited = x * (1 - self.alpha * noise_gate)
        return x_inhibited


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    """
    SegFormer Head with Optimized Architecture
    ASPP moved to C4 branch for speed (High Efficiency Mode)
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels



        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        # 这里的注意力模块保持在融合后，用于精修
        self.coord_att = CoordAttV2(embedding_dim, reduction=16, norm="gn")
        self.cbam = CBAM(embedding_dim, ratio=16)
        self.feat_inhibit = FeatureInhibition(embedding_dim)

        # 最后的分类头
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape


        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)



        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # 特征融合 (此时 C4 已经携带了 ASPP 的超强上下文信息)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # ==================== 后处理 (精修) ====================

        # 1. CoordAtt (定位)
        _c = self.coord_att(_c)

        # 2. CBAM (精修)
        _c = self.cbam(_c)

        # 3. Inhibition (去噪)
        _c = self.feat_inhibit(_c)


        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b0', pretrained=False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x