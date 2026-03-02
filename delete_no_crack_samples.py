import os
from PIL import Image
import numpy as np

# ================== 配置这里 ==================

# 原图所在文件夹（一般是 VOC 的 JPEGImages）
IMAGE_DIR = r"VOCdevkit/VOC2007/JPEGImages"

# 掩膜 PNG 所在文件夹（一般是 SegmentationClass）
MASK_DIR = r"VOCdevkit/VOC2007/SegmentationClass"

# 真正执行删除前先设为 False，只打印要删哪些文件，确认没问题再改成 True
DO_DELETE = True   # ⚠️ 改成 True 才会真的删文件！

# 前景面积比例阈值：
# mask 前景像素比例 < MIN_FG_RATIO 认为“无裂缝”，删
# mask 前景像素比例 > MAX_FG_RATIO 认为“大片区域”，删
MIN_FG_RATIO = 0.005    # 0.5% 以下当全黑 / 几乎没有
MAX_FG_RATIO = 0.05      # 5% 以上当大白块

# ===========================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def main():
    mask_files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")]
    mask_files.sort()
    print("在掩膜目录中找到 PNG 数量:", len(mask_files))

    delete_count = 0

    for name in mask_files:
        mask_path = os.path.join(MASK_DIR, name)
        mask = np.array(Image.open(mask_path))

        h, w = mask.shape[:2]
        total = h * w

        # 认为非 0 即前景（不管是 1 还是 255）
        fg_pixels = np.count_nonzero(mask)
        fg_ratio = fg_pixels / float(total)

        # 判断是否需要删除
        delete_flag = False
        reason = ""

        if fg_ratio < MIN_FG_RATIO:
            delete_flag = True
            reason = "无裂缝(几乎全黑)"
        elif fg_ratio > MAX_FG_RATIO:
            delete_flag = True
            reason = "大白块(前景比例 %.2f%%)" % (fg_ratio * 100)

        if not delete_flag:
            continue

        base = os.path.splitext(name)[0]

        # 找对应的原图（可能有多种后缀）
        img_paths = []
        for ext in IMG_EXTS:
            p = os.path.join(IMAGE_DIR, base + ext)
            if os.path.exists(p):
                img_paths.append(p)

        print(f"[删除候选] {name} -> {reason}")
        for p in img_paths:
            print("  对应原图:", p)

        if DO_DELETE:
            # 删除掩膜
            try:
                os.remove(mask_path)
                print("  已删除掩膜:", mask_path)
            except Exception as e:
                print("  删除掩膜出错:", e)

            # 删除原图
            for p in img_paths:
                try:
                    os.remove(p)
                    print("  已删除原图:", p)
                except Exception as e:
                    print("  删除原图出错:", e)

        delete_count += 1

    print("\n预计将删除（或已删除）样本数量:", delete_count)
    if not DO_DELETE:
        print("当前 DO_DELETE=False，只是预览。确认名单没问题后，把 DO_DELETE 改成 True 再运行一次。")


if __name__ == "__main__":
    main()
