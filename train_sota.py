import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import pandas as pd

# ------------------------------------------------------------------
# 导入你自己的模块 (确保 utils/dataloader.py 存在且包含 SegmentationDataset)
# ------------------------------------------------------------------
from nets.segformer import SegFormer
from nets.segformer_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

# 导入经典模型工厂 (需确保 nets/sota_models.py 存在)
try:
    from nets.sota_models import get_sota_model
except ImportError:
    print("错误：请先创建 nets/sota_models.py 文件！")
    exit()

# ==============================================================================
# SOTA 对比实验列表 (配置你想跑的所有模型)
# ==============================================================================
MODEL_COMPARISON_LIST = [


    # 2. 经典对比模型
    ("DeepLabV3+", "deeplabv3+", "resnet18"),

]


# ==============================================================================
# 主训练函数
# ==============================================================================
def run_training_for_model(model_display_name, model_type, backbone_name):
    print(f"\n{'=' * 60}")
    print(f">>>> 正在开始训练: {model_display_name} (Type: {model_type}, Backbone: {backbone_name})")
    print(f"{'=' * 60}\n")

    # --- 基础参数配置 ---
    Cuda = True
    seed = 11
    fp16 = True
    num_classes = 2  # 背景 + 裂缝

    # 预训练权重逻辑
    pretrained = True
    model_path = ""

    input_shape = [768, 768]

    # 训练周期设置
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    UnFreeze_Epoch = 100
    Unfreeze_batch_size =8  # 如果显存够大，可以改大一点
    Freeze_Train = True

    # 优化器设置
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2
    lr_decay_type = 'cos'

    # 保存与评估
    save_period = 10
    save_dir = os.path.join('logs_sota', model_display_name)

    eval_flag = True
    eval_period = 5

    # ---------------------------------------------------
    # [核心修改] 这里指定你的 VOC 数据集路径
    # ---------------------------------------------------
    VOCdevkit_path = 'VOCdevkit'

    dice_loss = True
    focal_loss = True
    cls_weights = np.array([1.0, 5.0], np.float32)
    num_workers = 4

    # --- 初始化环境 ---
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    rank = 0

    # --- 构建模型 ---
    if model_type == "hais":
        model = SegFormer(num_classes=num_classes, phi=backbone_name, pretrained=pretrained)
        if pretrained and model_path == "":
            download_weights(backbone_name)
    else:
        model = get_sota_model(model_type, num_classes=num_classes, encoder=backbone_name)

    if not pretrained:
        weights_init(model)

    # --- 记录器 ---
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    # --- 混合精度 ---
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # =======================================================================
    # [核心修改] 读取数据集 (完全按照你给的代码段)
    # =======================================================================
    print("正在加载数据集 (Reading VOC format)...")

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    print(f"训练集数量: {num_train}, 验证集数量: {num_val}")

    # --- 冻结逻辑 ---
    if Freeze_Train:
        if model_type == "hais":
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            if hasattr(model, 'encoder'):
                for param in model.encoder.parameters(): param.requires_grad = False
            elif hasattr(model, 'backbone'):
                for param in model.backbone.parameters(): param.requires_grad = False

    # =======================================================================
    # [核心修改] 使用 SegmentationDataset 构建 DataLoader
    # =======================================================================
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    # 这里直接使用 utils.dataloader 里的标准类
    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=seg_dataset_collate,
                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=seg_dataset_collate,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    # --- Eval 回调 ---
    eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                 eval_flag=eval_flag, period=eval_period)

    # --- 开始训练循环 ---
    UnFreeze_flag = False

    # 动态学习率计算
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0:
        raise ValueError("数据集太小，无法训练，请检查 VOCdevkit 路径或 batch_size")

    # -------------------------------------------------------------------
    # Epoch Loop
    # -------------------------------------------------------------------
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 解冻阶段
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            # 重新计算 LR
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            # 解冻参数
            if model_type == "hais":
                for param in model.backbone.parameters(): param.requires_grad = True
            elif hasattr(model, 'encoder'):
                for param in model.encoder.parameters(): param.requires_grad = True
            elif hasattr(model, 'backbone'):
                for param in model.backbone.parameters(): param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            # 重新加载 DataLoader (batch_size 变了)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=seg_dataset_collate,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=seg_dataset_collate,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

            UnFreeze_flag = True
            print("Start Unfreeze Training...")

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                      gen, gen_val, UnFreeze_Epoch, Cuda, \
                      dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

    return eval_callback


# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":
    final_results = []

    # 确保 VOCdevkit 存在
    if not os.path.exists("VOCdevkit"):
        print("Error: 当前目录下找不到 'VOCdevkit' 文件夹，请检查路径！")
        exit()

    for display_name, m_type, backbone in MODEL_COMPARISON_LIST:
        torch.cuda.empty_cache()

        try:
            callback = run_training_for_model(display_name, m_type, backbone)
            final_results.append({
                "Model": display_name,
                "Status": "Success",
                "Log Dir": os.path.join("logs_sota", display_name)
            })

        except Exception as e:
            print(f"模型 {display_name} 训练失败: {e}")
            import traceback

            traceback.print_exc()
            final_results.append({"Model": display_name, "Status": "Failed"})

    print("\n========== 所有对比实验运行结束 ==========")
    df = pd.DataFrame(final_results)
    print(df)