import torch
from thop import profile
from nets.segformer import SegFormer  # 确保这里能导入您修改后的模型类

# 1. 实例化您的模型 (确保 num_classes 和 phi 与训练时一致)
# 如果您改了类名为 UCASegFormer，请相应修改
model = SegFormer(num_classes=2, phi='b0')

# 2. 创建一个伪造的输入张量 (模拟一张图片输入)
# 输入尺寸必须是您训练/推理时的大小，通常是 512x512
input_tensor = torch.randn(1, 3, 512, 512)

# 3. 计算 FLOPs 和 Params
flops, params = profile(model, inputs=(input_tensor,))

print(f"=============================================")
print(f"Input Shape: {input_tensor.shape}")
print(f"FLOPs: {flops / 1e9:.3f} G (Giga FLOPs)")
print(f"Params: {params / 1e6:.3f} M (Million Params)")
print(f"=============================================")