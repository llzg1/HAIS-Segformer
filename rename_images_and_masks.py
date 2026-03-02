import os
import numpy as np

# TODO: 根据你自己的路径修改这里
IMG_DIR  = r"VOCdevkit\VOC2007\JPEGImages"
MASK_DIR = r"VOCdevkit\VOC2007\SegmentationClass"

IMG_EXTS  = (".jpg", ".jpeg", ".png", ".bmp")
MASK_EXTS = (".png", ".jpg", ".jpeg")

def main():
    # 读取并排序所有图片文件
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(IMG_EXTS)]
    img_files.sort()

    if not img_files:
        print("JPEGImages 里没有找到图片。")
        return

    print("找到图片数量：", len(img_files))

    # 先建立 “原始名字(不含后缀) -> 新名字(不含后缀)” 的映射
    name_map = {}  # old_stem -> new_stem
    for idx, fname in enumerate(img_files, 1):
        stem, _ = os.path.splitext(fname)
        new_stem = f"{idx:03d}"
        name_map[stem] = new_stem

    # 第一步：给图片改成临时名，避免冲突
    tmp_names = []
    for idx, fname in enumerate(img_files, 1):
        old_path = os.path.join(IMG_DIR, fname)
        _, ext = os.path.splitext(fname)
        tmp_name = f"tmp_{idx:03d}{ext.lower()}"
        tmp_path = os.path.join(IMG_DIR, tmp_name)
        os.rename(old_path, tmp_path)
        tmp_names.append((tmp_name, ext.lower()))

    # 第二步：再从临时名改成最终名 001.jpg 002.jpg ...
    for idx, (tmp_name, ext) in enumerate(tmp_names, 1):
        tmp_path = os.path.join(IMG_DIR, tmp_name)
        new_name = f"{idx:03d}{ext}"
        new_path = os.path.join(IMG_DIR, new_name)
        os.rename(tmp_path, new_path)

    print("图片重命名完成。")

    # 第三步：按映射重命名 SegmentationClass 下的掩膜
    missing_masks = []
    for old_stem, new_stem in name_map.items():
        # 在 MASK_DIR 里找对应的 old_stem.xxx
        mask_old_path = None
        for ext in MASK_EXTS:
            candidate = os.path.join(MASK_DIR, old_stem + ext)
            if os.path.exists(candidate):
                mask_old_path = candidate
                break

        if mask_old_path is None:
            missing_masks.append(old_stem)
            continue

        _, mask_ext = os.path.splitext(mask_old_path)
        mask_new_path = os.path.join(MASK_DIR, new_stem + mask_ext.lower())
        os.rename(mask_old_path, mask_new_path)

    if missing_masks:
        print("有这些图片在 SegmentationClass 中没找到对应的掩膜：")
        print(missing_masks)
    else:
        print("所有掩膜也已按顺序重命名完成。")

    print("全部完成！记得之后重新运行一次 voc_annotation.py 生成新的 train/val 列表。")

if __name__ == "__main__":
    main()
