import torch
import os
import time
import pandas as pd
from thop import profile
from nets.segformer_comparative import SegFormer

# ================= 配置区域 =================
phi = 'b0'
num_classes = 2
input_shape = [1024,1024]

# FPS 测试循环次数 (越多越准)
fps_repeat_times = 500

# 定义消融实验全家桶
ablation_configs = [
    # --- 1. 基准 ---
    ('Baseline (B0)', 'none'),

    # --- 2. 单模块消融 ---
    ('B0 + CoordAtt (CA)', 'coord'),
    ('B0 + CBAM', 'cbam'),
    ('B0 + FIM', 'fim_only'),

    # --- 3. 两两组合消融 ---
    ('B0 + Hybrid (CA+CBAM)', 'hybrid'),
    ('B0 + CA + FIM', 'ca_fim'),
    ('B0 + CBAM + FIM', 'cbam_fim'),

    # --- 4. 完整版 ---
    ('HAIS (Ours)', 'hais')
]


# ===========================================

def calculate_metrics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-----------------------------------------------------------------------")
    print(f"全能消融指标计算 (Backbone: {phi} | Input: {input_shape} | Device: {torch.cuda.get_device_name(0)})")
    print(f"-----------------------------------------------------------------------")

    results = []
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    for name, att_type in ablation_configs:
        try:
            print(f"正在测试: {name} ...", end="\r")

            # 1. 实例化模型
            model = SegFormer(num_classes=num_classes, phi=phi, pretrained=False, att_type=att_type)
            model.eval()
            model.to(device)

            # --------------------------
            # 指标 A: 参数量 (Params)
            # --------------------------
            total_params = sum(p.numel() for p in model.parameters())
            params_m = total_params / 1e6

            # --------------------------
            # 指标 B: 计算量 (GFLOPs)
            # --------------------------
            # thop 不支持 fp16，所以用 fp32 算 GFLOPs
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            gflops = flops / 1e9

            # --------------------------
            # 指标 C: 模型大小 (Size)
            # --------------------------
            temp_name = "temp_weights_measure.pth"
            torch.save(model.state_dict(), temp_name)
            file_size_mb = os.path.getsize(temp_name) / 1024 / 1024
            os.remove(temp_name)

            # --------------------------
            # 指标 D: FPS (推理速度)
            # --------------------------
            # D.1 预热 (Warm-up)
            with torch.no_grad():
                for _ in range(50):
                    _ = model(dummy_input)

            # D.2 正式测速 (开启 FP16 半精度加速，模拟 3090 真实性能)
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                for _ in range(fps_repeat_times):
                    with torch.cuda.amp.autocast(enabled=True):
                        _ = model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.time()

            avg_time = (end_time - start_time) / fps_repeat_times
            fps = 1 / avg_time

            # --------------------------
            # 记录结果
            # --------------------------
            results.append({
                "Model Configuration": name,
                "att_type": att_type,
                "Params (M)": round(params_m, 2),
                "GFLOPs (G)": round(gflops, 2),
                "Size (MB)": round(file_size_mb, 2),
                "FPS": round(fps, 1)  # FPS 保留一位小数即可
            })
            print(f"[OK] {name}: {fps:.1f} FPS")

        except Exception as e:
            print(f"\n[Error] {name} ({att_type}): {str(e)}")

    print("\n\n========================= 消融实验全指标汇总表 =========================")
    df = pd.DataFrame(results)
    try:
        print(df.to_markdown(index=False))
    except:
        print(df)
    print("======================================================================")


if __name__ == "__main__":
    calculate_metrics()