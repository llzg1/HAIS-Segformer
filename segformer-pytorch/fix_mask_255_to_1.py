import os
import numpy as np
from PIL import Image

# 标签所在目录
seg_dir = r"VOCdevkit/VOC2007/SegmentationClass"

for name in os.listdir(seg_dir):
    if not name.lower().endswith(".png"):
        continue

    path = os.path.join(seg_dir, name)
    img = Image.open(path)

    # 转成单通道灰度
    img = img.convert("L")
    arr = np.array(img)

    # 查看当前标签中有哪些像素值
    unique_vals = np.unique(arr)
    print(name, "unique values:", unique_vals)

    # 把 255 改成 1，其它不是 1 的都归为 0（保证只有 0 和 1）
    arr[arr == 255] = 1
    arr[arr != 1] = 0

    # 保存回原文件（覆盖）
    Image.fromarray(arr.astype("uint8")).save(path)

print("Done: 255 -> 1, 已处理完所有 png 标签。")
