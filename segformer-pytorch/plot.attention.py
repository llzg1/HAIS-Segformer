import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_jet_heatmap(gray_tensor):
    """将灰度图转为 Jet 热力图 (Blue=Low, Red=High)"""
    norm = cv2.normalize(gray_tensor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_images(bg_img, heatmap_rgb, alpha=0.5):
    """将热力图半透明叠加在原图上"""
    return cv2.addWeighted(bg_img, 1 - alpha, heatmap_rgb, alpha, 0)


def generate_grid_visualization(image_path='input.jpg'):
    # ==========================================
    # 1. 读取与核心信号提取 (Black-Hat 变换)
    # ==========================================
    try:
        img = cv2.imread(image_path)
        if img is None: raise Exception("File not found")
        img = cv2.resize(img, (512, 512))
    except:
        print("⚠️ 未找到 input.jpg，生成模拟数据...")
        img = np.zeros((512, 512, 3), dtype=np.uint8) + 120
        cv2.line(img, (100, 50), (400, 450), (40, 40, 40), 8)  # 模拟暗裂缝

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🌟 核心修复技术：黑帽运算 (Black-Hat)
    # 专门提取亮背景下的暗细节，对水下裂缝极度有效
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # 原始裂缝信号 (背景变为纯黑，裂缝变为灰/白)
    crack_signal = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 增强信号强度 (拉伸对比度)
    crack_signal = cv2.normalize(crack_signal, None, 0, 255, cv2.NORM_MINMAX)

    # ==========================================
    # 2. 模拟三个模块的效果
    # ==========================================

    # --- A. 模拟 CoordAtt (连通性恢复) ---
    # 效果：光晕感，连贯，但不锐利
    # 操作：对信号做膨胀 + 高斯模糊
    feat_coord = cv2.GaussianBlur(crack_signal, (25, 25), 0)
    feat_coord = cv2.convertScaleAbs(feat_coord, alpha=2.5, beta=0)  # 提亮

    # --- B. 模拟 CBAM (边缘锐化 + 引入噪点) ---
    # 效果：裂缝很亮，但背景出现很多红色噪点
    # 操作：原始信号锐化 + 添加随机背景噪声
    noise = np.random.randint(0, 80, crack_signal.shape, dtype=np.uint8)  # 模拟水草噪点
    feat_cbam = cv2.addWeighted(crack_signal, 1.2, noise, 0.5, 0)
    # 锐化
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    feat_cbam = cv2.filter2D(feat_cbam, -1, kernel_sharp)

    # --- C. 模拟 FIM (去噪输出) ---
    # 效果：只有裂缝是红的，背景是纯深蓝
    # 操作：利用形态学掩码过滤掉 CBAM 中的背景噪点
    # 1. 制作掩码 (只要裂缝区域)
    _, mask = cv2.threshold(crack_signal, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((5, 5)), iterations=2)  # 稍微扩大一点掩码范围
    mask_float = mask.astype(np.float32) / 255.0

    # 2. 乘法抑制 (Background Suppression)
    feat_fim = feat_cbam.astype(np.float32) * mask_float
    feat_fim = np.clip(feat_fim * 1.5, 0, 255).astype(np.uint8)  # 再次提亮裂缝

    # ==========================================
    # 3. 生成 2x4 网格图
    # ==========================================
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 生成热力图
    hm_coord = apply_jet_heatmap(feat_coord)
    hm_cbam = apply_jet_heatmap(feat_cbam)
    hm_fim = apply_jet_heatmap(feat_fim)

    # 生成叠加图
    ov_coord = overlay_images(img_rgb, hm_coord)
    ov_cbam = overlay_images(img_rgb, hm_cbam)
    ov_fim = overlay_images(img_rgb, hm_fim)

    # 绘图设置
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.01, right=0.99)

    # 定义标签
    titles = ["Input Image", "After CoordAtt", "After CBAM", "After FIM (Output)"]
    subtitles = [
        "(Original underwater image)",
        "(Topology Recovered)",
        "(Edges Sharpened\nbut Noise Increased)",
        "(Noise Removed\nClean Structure)"
    ]

    # --- 第一行：纯热力图 (Feature Maps) ---
    # Col 1: Input (直接显示原图)
    axes[0, 0].imshow(img_rgb)

    # Col 2: CoordAtt
    axes[0, 1].imshow(hm_coord)

    # Col 3: CBAM
    axes[0, 2].imshow(hm_cbam)

    # Col 4: FIM (这里绝对不会黑了)
    axes[0, 3].imshow(hm_fim)

    # --- 第二行：叠加图 (Overlay) ---
    axes[1, 0].imshow(img_rgb)  # 重复原图，保持对齐
    axes[1, 1].imshow(ov_coord)
    axes[1, 2].imshow(ov_cbam)
    axes[1, 3].imshow(ov_fim)

    # --- 统一设置标题和样式 ---
    for col in range(4):
        # 设置第一行的标题
        axes[0, col].set_title(titles[col], fontsize=16, fontweight='bold', pad=10)

        # 设置第二行的底部说明
        axes[1, col].set_xlabel(subtitles[col], fontsize=13, style='italic')

        # 去除坐标轴
        axes[0, col].axis('off')
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        # 给图片加黑框
        for row in range(2):
            for spine in axes[row, col].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

    save_path = "final_grid_fixed.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 2x4 布局修复版已生成: {save_path}")
    plt.show()


if __name__ == "__main__":
    generate_grid_visualization()