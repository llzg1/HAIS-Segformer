# eval_underwater.py
import os
from PIL import Image
from tqdm import tqdm

from segformer import SegFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results


def eval_one_model(
    model_path,
    phi="b0",
    image_set="test",                # "train" / "val" / "test" 都可以
    voc_root="VOCdevkit",
    miou_out_dir="miou_out/ca_b0_test",
):
    """
    在给定的 image_set 上，对一个 segformer 模型做 mIoU/mPA/Acc 评估。
    """
    # 你的语义分割类别：0=背景，1=裂缝
    num_classes = 2
    name_classes = ["background", "crack"]

    # 读取 image_ids 列表
    image_set_path = os.path.join(voc_root, "VOC2007", "ImageSets", "Segmentation", f"{image_set}.txt")
    with open(image_set_path, "r") as f:
        image_ids = [x.strip() for x in f.readlines()]

    gt_dir = os.path.join(voc_root, "VOC2007", "SegmentationClass")
    pred_dir = os.path.join(miou_out_dir, "detection-results")
    os.makedirs(pred_dir, exist_ok=True)

    # 1. 载入模型
    print(f"\n========== Evaluating [{os.path.basename(model_path)}] on [{image_set}] ==========")
    print("Load model...")
    segformer = SegFormer_Segmentation(
        model_path=model_path,
        num_classes=num_classes,      # 注意：这里覆盖 segformer.py 里的 _defaults
        phi=phi,
    )
    print("Load model done.")

    # 2. 逐张生成预测 PNG（灰度图，每个像素是类别 id）
    print("Get predict results...")
    for image_id in tqdm(image_ids):
        img_path = os.path.join(voc_root, "VOC2007", "JPEGImages", image_id + ".jpg")
        image = Image.open(img_path)
        pred = segformer.get_miou_png(image)   # 内置的专用函数
        pred.save(os.path.join(pred_dir, image_id + ".png"))
    print("Get predict results done.")

    # 3. 计算 mIoU / mPA / Acc 等指标
    print("Get miou...")
    hist, IoUs, PA_Recall, Precision = compute_mIoU(
        gt_dir, pred_dir, image_ids, num_classes, name_classes
    )
    print("Get miou done.")

    # 会自动把每类 IoU、混淆矩阵画到 miou_out_dir 里
    show_results(miou_out_dir, hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == "__main__":
    # 这里填你想评估的权重路径（在 logs 目录下）
    model_path = r"logs/segformer_b0_ca_best_mIoU71_15.pth"

    # 评估 test 集
    eval_one_model(
        model_path=model_path,
        phi="b0",
        image_set="test",
        voc_root="VOCdevkit",
        miou_out_dir="miou_out/ca_b0_test",
    )
