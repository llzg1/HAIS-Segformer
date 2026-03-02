import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


# ==============================================================================
# Part 1: 基础组件 (保持不变)
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsigmoid = HSigmoid()

    def forward(self, x):
        return x * self.hsigmoid(x)


# ==============================================================================
# Part 2: 你的核心模块 (HAIS Components) - [核心修改区域]
# ==============================================================================

# 2.1 CoordAtt (保持不变)
class CoordAttV2(nn.Module):
    def __init__(self, inp, reduction=16, norm="gn"):
        super().__init__()
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.GroupNorm(8 if mip % 8 == 0 else 1, mip) if norm == "gn" else nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.norm(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


# 2.2 CBAM (保持不变)
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
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(7)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# 2.3 Feature Inhibition Module (FIM) -> [修改: 改为 Soft-FIM]
class FeatureInhibition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 使用 1x1 卷积生成抑制掩码
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        noise_gate = self.gate_conv(x)
        # [核心修改] 柔性抑制: 不是完全减去，而是乘以 (1 - mask * 0.5)
        # 0.5 是衰减系数，防止把细小裂痕当成背景完全抹除
        return x * (1 - noise_gate * 0.5)


# 2.4 [新增] HAIS_V2 Parallel Module (并行+融合)
class HAIS_Parallel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1. 两个并行分支
        self.coord = CoordAttV2(channels)  # 负责精准定位 (X, Y)
        self.cbam = CBAM(channels, ratio=16)  # 负责上下文纹理 (Context)

        # 2. 抑制模块
        self.fim = FeatureInhibition(channels)  # 负责清洗背景

        # 3. 可学习的融合权重 (初始化为 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # 4. 融合后的特征整合 (可选，增加非线性)
        self.fuse_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        identity = x

        # --- 并行计算 ---
        feat_coord = self.coord(x)
        feat_cbam = self.cbam(x)

        # --- 自适应加权融合 ---
        # 这是一个简单的加权和，让网络自己学 alpha 和 beta
        feat_fused = (self.alpha * feat_coord) + (self.beta * feat_cbam)

        # --- 背景抑制 ---
        feat_clean = self.fim(feat_fused)

        # --- 残差连接 (Fusion后的特征 + 原始输入) ---
        return identity + self.fuse_conv(feat_clean)


# ==============================================================================
# Part 3: 对比实验模块库 (保持不变，省略部分重复代码)
# ==============================================================================
# (此处保留你原代码中的 SEBlock, ECABlock, TripletAttention, ShuffleAttention,
#  BAM, SimAM, SKAttention, ContextBlock, DropBlock, SoftThresholding 等)
# 为了节省篇幅，假设这里和你原来的一样
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class TripletAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cw = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.hc = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = x_perm1 * self.cw(
            torch.cat((torch.max(x_perm1, 1)[0].unsqueeze(1), torch.mean(x_perm1, 1).unsqueeze(1)), dim=1))
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = x_perm2 * self.hc(
            torch.cat((torch.max(x_perm2, 1)[0].unsqueeze(1), torch.mean(x_perm2, 1).unsqueeze(1)), dim=1))
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out3 = x * self.spatial(torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1))
        return 1 / 3 * (x_out1 + x_out2 + x_out3)


class ShuffleAttention(nn.Module):
    def __init__(self, channel=512, groups=64):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w).permute(0, 2, 1, 3, 4).reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.channel_shuffle(x, self.groups)
        group_c = c // self.groups
        x_split = x.reshape(b, self.groups, group_c, h, w)
        x_channel, x_spatial = x_split.chunk(2, dim=2)
        x_channel_out = (x_channel * self.sigmoid(
            self.cweight * x_channel.mean(dim=[3, 4], keepdim=True) + self.cbias)).reshape(b, -1, h, w)
        x_spatial_out = (x_spatial * self.sigmoid(
            self.sweight * self.gn(x_spatial.reshape(b * self.groups, -1, h, w)).reshape(b, self.groups, -1, h,
                                                                                         w) + self.sbias)).reshape(b,
                                                                                                                   -1,
                                                                                                                   h, w)
        return self.channel_shuffle(torch.cat([x_channel_out, x_spatial_out], dim=1), 2)


class BAM(nn.Module):
    def __init__(self, gate_channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(nn.Linear(gate_channel, gate_channel // reduction), nn.ReLU(),
                                         nn.Linear(gate_channel // reduction, gate_channel))
        self.spatial_att = nn.Sequential(nn.Conv2d(gate_channel, gate_channel // reduction, 1),
                                         nn.BatchNorm2d(gate_channel // reduction), nn.ReLU(),
                                         nn.Conv2d(gate_channel // reduction, gate_channel // reduction, 3, padding=4,
                                                   dilation=4), nn.BatchNorm2d(gate_channel // reduction), nn.ReLU(),
                                         nn.Conv2d(gate_channel // reduction, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        att = 1 + self.sigmoid(self.channel_att(self.avg_pool(x).view(b, c)).view(b, c, 1, 1) + self.spatial_att(x))
        return x * att


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class SKAttention(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features / r), L)
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(features, features, 3, stride, 1 + i, dilation=1 + i, groups=G, bias=False),
            nn.BatchNorm2d(features), nn.ReLU()) for i in range(M)])
        self.fc = nn.Sequential(nn.Conv2d(features, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU())
        self.fcs = nn.ModuleList([nn.Conv2d(d, features, 1) for i in range(M)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)
        U = torch.sum(feats, dim=1)
        Z = self.fc(torch.mean(U, dim=[2, 3], keepdim=True))
        weights = self.softmax(torch.stack([fc(Z) for fc in self.fcs], dim=1))
        return torch.sum(weights * feats, dim=1)


class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio=0.25):
        super().__init__()
        planes = int(inplanes * ratio)
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1), nn.LayerNorm([planes, 1, 1]),
                                              nn.ReLU(inplace=True), nn.Conv2d(planes, inplanes, kernel_size=1))

    def forward(self, x):
        b, c, h, w = x.size()
        mask = self.softmax(self.conv_mask(x).view(b, 1, h * w)).unsqueeze(3)
        context = torch.matmul(x.view(b, c, h * w).unsqueeze(1), mask).view(b, c, 1, 1)
        return x + self.channel_add_conv(context)


class SoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // 16), nn.ReLU(), nn.Linear(channels // 16, channels),
                                nn.Sigmoid())

    def forward(self, x):
        threshold = self.fc(self.global_pool(torch.abs(x)).view(x.shape[0], -1)).view(x.shape[0], -1, 1, 1) * torch.abs(
            x)
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class DropBlock(nn.Module):
    def __init__(self, block_size=5, keep_prob=0.9):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training: return x
        gamma = (1. - self.keep_prob) / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)
        mask = 1. - F.max_pool2d(mask.unsqueeze(1), self.block_size, 1, self.block_size // 2)
        return x * mask * (mask.numel() / mask.sum())


# ==============================================================================
# Part 4: SegFormer Head (Central Control) - [核心修改区域]
# ==============================================================================
class SegFormerHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768,
                 dropout_ratio=0.1, att_type='hais'):
        super(SegFormerHead, self).__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels
        self.linear_c4 = MLP(c4_in, embedding_dim)
        self.linear_c3 = MLP(c3_in, embedding_dim)
        self.linear_c2 = MLP(c2_in, embedding_dim)
        self.linear_c1 = MLP(c1_in, embedding_dim)
        self.linear_fuse = ConvModule(c1=embedding_dim * 4, c2=embedding_dim, k=1)

        # ---------------- 核心切换逻辑 ----------------
        self.att_type = att_type
        self.attention = nn.Identity()
        self.inhibition = nn.Identity()

        # [Group 1: Position/Global]
        if att_type == 'se':
            self.attention = SEBlock(embedding_dim)
        elif att_type == 'eca':
            self.attention = ECABlock(embedding_dim)
        elif att_type == 'triplet':
            self.attention = TripletAttention()
        elif att_type == 'shuffle':
            self.attention = ShuffleAttention(embedding_dim)
        elif att_type == 'coord':
            self.attention = CoordAttV2(embedding_dim)

        # [Group 2: Hybrid/Refinement]
        elif att_type == 'bam':
            self.attention = BAM(embedding_dim)
        elif att_type == 'simam':
            self.attention = SimAM()
        elif att_type == 'sk':
            self.attention = SKAttention(embedding_dim)
        elif att_type == 'gcnet':
            self.attention = ContextBlock(embedding_dim)
        elif att_type == 'cbam':
            self.attention = CBAM(embedding_dim)

        # [Group 3: Inhibition]
        elif att_type == 'dropblock':
            self.inhibition = DropBlock(block_size=5)
        elif att_type == 'soft_thresh':
            self.inhibition = SoftThresholding(embedding_dim)
        elif att_type == 'spatial_drop':
            self.inhibition = nn.Dropout2d(p=0.3)

        # [消融实验组合]
        elif att_type == 'hybrid':  # 串行模式 (旧)
            self.attention = nn.Sequential(CoordAttV2(embedding_dim), CBAM(embedding_dim))
            self.inhibition = nn.Identity()
        elif att_type == 'fim_only':
            self.attention = nn.Identity()
            self.inhibition = FeatureInhibition(embedding_dim)

        # [HAIS: Ours] -> 升级为 HAIS-V2 并行版
        elif att_type == 'hais':
            # 注意：新的 HAIS_Parallel 已经包含了注意力+融合+抑制
            # 所以这里只赋值给 attention，把 inhibition 设为空，防止重复处理
            self.attention = HAIS_Parallel(embedding_dim)
            self.inhibition = nn.Identity()
            # ---------------------------------------------

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        _c4 = F.interpolate(self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]),
                            size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]),
                            size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]),
                            size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # 核心增强
        _c = self.attention(_c)  # 如果是 HAIS，这里已经完成了全部并行融合与抑制
        _c = self.inhibition(_c)  # 如果是单模块消融，这里执行抑制

        x = self.linear_pred(self.dropout(_c))
        return x


# ==============================================================================
# Part 5: Main SegFormer Model (保持不变)
# ==============================================================================
class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b0', pretrained=False, att_type='hais'):
        super(SegFormer, self).__init__()
        self.in_channels = \
            {'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512], 'b3': [64, 128, 320, 512],
             'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512]}[phi]
        self.backbone = {'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2, 'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5}[phi](
            pretrained)
        self.embedding_dim = {'b0': 256, 'b1': 256, 'b2': 768, 'b3': 768, 'b4': 768, 'b5': 768}[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim, att_type=att_type)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x