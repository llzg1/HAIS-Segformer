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

# 导入你自己的模块
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
# 格式: (显示名称, 模型类型标识, Backbone/参数)
MODEL_COMPARISON_LIST = [
    # 1. 你的 HAIS 模型
    ("HAIS-SegFormer", "hais", "b0"),

    # 2. 经典对比模型 (Backbone 选 resnet34/18 以对标 B0)
    ("DeepLabV3+", "deeplabv3+", "resnet34"),
    ("U-Net", "unet", "resnet34"),
    ("PSPNet", "pspnet", "resnet34"),
    # ("SegNet",         "segnet",     "resnet34"), # 可选
]


# ==============================================================================
# 主训练函数 (封装原本的训练逻辑)
# ==============================================================================
def run_training_for_model(model_display_name, model_type, backbone_name):
    print(f"\n{'=' * 60}")
    print(f">>>> 正在开始训练: {model_display_name} (Type: {model_type}, Backbone: {backbone_name})")
    print(f"{'=' * 60}\n")

    # --- 原有参数配置 (保持不变) ---
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = True
    num_classes = 2

    # 预训练权重逻辑
    pretrained = True  # 经典模型通常都加载 ImageNet 权重
    model_path = ""  # 对比实验通常从 ImageNet 预训练开始，而不是加载你自己的 .pth

    input_shape = [512, 512]  # 建议统一为 512x512，768 可能爆显存

    # 训练周期
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 8  # 显存不够改小
    Freeze_Train = True

    # 优化器
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2
    lr_decay_type = 'cos'

    # 保存与评估
    save_period = 10  # 减少保存频率，省硬盘
    # ！！！关键修改：每个模型保存到不同文件夹！！！
    save_dir = os.path.join('logs_sota', model_display_name)

    eval_flag = True
    eval_period = 5
    VOCdevkit_path = 'VOCdevkit'
    dice_loss = True
    focal_loss = True
    cls_weights = np.array([1.0, 5.0], np.float32)  # 你的权重
    num_workers = 4

    # --- 初始化环境 ---
    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    rank = 0

    # --- 构建模型 (Model Selection) ---
    if model_type == "hais":
        # 你的模型
        model = SegFormer(num_classes=num_classes, phi=backbone_name, pretrained=pretrained)
        # 加载 B0 预训练权重 (如果需要)
        if pretrained and model_path == "":
            download_weights(backbone_name)  # 这里原本是 'b0'
    else:
        # SOTA 模型 (自动加载 ImageNet 权重)
        model = get_sota_model(model_type, num_classes=num_classes, encoder=backbone_name)

    if not pretrained:
        weights_init(model)

    # --- 记录器初始化 ---
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

    # --- 读取数据集 ---
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # --- 主干冻结逻辑 (SegFormer 特有，SOTA 模型简单处理) ---
    # 如果是经典模型，我们通常也冻结 Encoder
    if Freeze_Train:
        if model_type == "hais":
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            # 对于 smp 模型，encoder 是 model.encoder
            if hasattr(model, 'encoder'):
                for param in model.encoder.parameters():
                    param.requires_grad = False
            # 对于 torchvision 模型，可能是 model.backbone
            elif hasattr(model, 'backbone'):
                for param in model.backbone.parameters():
                    param.requires_grad = False

    # --- 准备 DataLoader ---
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

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

    # 动态调整学习率策略
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
        raise ValueError("数据集太小，无法训练")

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

            # 重新加载 DataLoader
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

    # 训练结束，返回最后一次 Eval 的结果（或者你可以修改 EvalCallback 让它返回最佳值）
    # 这里我们简单读取 EvalCallback 日志里的最佳值（如果有记录的话）
    # 为了简化，我们假设 EvalCallback 已经把最好的结果保存在 logs 里了

    return eval_callback  # 返回回调对象，后续可以提取数据


# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":
    final_results = []

    for display_name, m_type, backbone in MODEL_COMPARISON_LIST:
        # 清理显存，防止 OOM
        torch.cuda.empty_cache()

        try:
            callback = run_training_for_model(display_name, m_type, backbone)

            # 这里简单记录完成状态，具体 mIoU 需要你去 logs 文件夹里的 epoch_map.txt 查看
            # 或者你可以修改 EvalCallback 让它有一个 self.best_miou 属性
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