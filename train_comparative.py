# ------------------------------------------------------------------
# train_comparative.py
# 用于 HAIS-SegFormer 的消融实验与对比实验
# ------------------------------------------------------------------
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

# ==================================================================
# 【关键】导入支持对比实验的 SegFormer 类
# ==================================================================
from nets.segformer_comparative import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ------------------------------------------------------------------
    #   Cuda: 是否使用 GPU
    # ------------------------------------------------------------------
    Cuda = True

    # ------------------------------------------------------------------
    #   distributed: 单机多卡设置 (单卡3090设为 False)
    # ------------------------------------------------------------------
    distributed = False

    # ------------------------------------------------------------------
    #   fp16: 混合精度训练 (3090 必须开，提速省显存)
    # ------------------------------------------------------------------
    fp16 = True

    # ------------------------------------------------------------------
    #   num_classes: 数据集类别 (背景 + 裂缝 = 2)
    # ------------------------------------------------------------------
    num_classes = 2

    # ------------------------------------------------------------------
    #   phi: 骨干网络 (B0~B5, 你的论文选定 B0
    # ------------------------------------------------------------------
    phi = "b0"

    # ==================================================================
    # 【核心控制台】 实验变量设置
    # ==================================================================
    # att_type: 决定本次运行的网络结构
    # 可选值:
    #   --- 你的方法 ---
    #   'hais'        : 完整版 (CoordAtt + CBAM + FIM)
    #
    #   --- 第一组对比 (VS CoordAtt) ---
    #   'se'          : SE Block
    #   'eca'         : ECA Block
    #   'triplet'     : Triplet Attention
    #   'shuffle'     : Shuffle Attention
    #   'coord'       : CoordAtt Only
    #
    #   --- 第二组对比 (VS CBAM) ---
    #   'bam'         : BAM
    #   'simam'       : SimAM
    #   'sk'          : SK Attention
    #   'gcnet'       : GCNet
    #   'cbam'        : CBAM Only
    #
    #   --- 第三组对比 (VS FIM) ---
    #   'soft_thresh' : Soft Thresholding (去噪)
    #   'dropblock'   : DropBlock
    #   'spatial_drop': Spatial Dropout
    # ==================================================================
    att_type = "spatial_drop"  # <--- 每次跑实验改这里！！！

    # ------------------------------------------------------------------
    #   input_shape: 输入分辨率 (论文设定 768x768)
    # ------------------------------------------------------------------
    input_shape = [768, 768]

    # ------------------------------------------------------------------
    #   model_path: 预训练权重路径
    #   注意: 即使 Head 结构变了，这里依然加载官方的 b0 权重，
    #   脚本会自动过滤掉不匹配的 Head 部分，只加载 Backbone。
    # ------------------------------------------------------------------
    model_path = "model_data/segformer_b0_weights_voc.pth"
    pretrained = False  # 因为我们手动加载 model_path，所以这里设 False

    # ------------------------------------------------------------------
    #   训练参数设置 (针对 3090 优化)
    # ------------------------------------------------------------------
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8  # 3090 显存大，可以设 8 或 16

    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 8  # 768分辨率下，8 是比较安全的，显存够可尝试 12

    Freeze_Train = True  # 建议冻结训练，保护 Backbone 特征

    # 优化器参数
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2
    lr_decay_type = 'cos'

    # ==================================================================
    #   日志保存路径 (自动根据 att_type 命名，防止覆盖)
    # ==================================================================
    save_period = 5
    save_dir = os.path.join('logs', f'{att_type}_{phi}')

    # ------------------------------------------------------------------
    #   数据集路径
    # ------------------------------------------------------------------
    VOCdevkit_path = 'VOCdevkit'

    # ==================================================================
    # 【Loss Tricks】 论文中提到的混合损失与权重
    # ==================================================================
    dice_loss = True
    focal_loss = True
    # 背景权重 1.0，裂缝权重 5.0
    cls_weights = np.array([1.0, 5.0], np.float32)

    # 其他
    eval_flag = True
    eval_period = 5
    num_workers = 8
    seed = 11
    seed_everything(seed)

    # ------------------------------------------------------------------
    #   设备设置
    # ------------------------------------------------------------------
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    # ------------------------------------------------------------------
    #   模型初始化
    # ------------------------------------------------------------------
    # 这里将 att_type 传入模型
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained, att_type=att_type)

    if not pretrained:
        weights_init(model)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------
        #   智能加载权重 (允许 Head 部分不匹配)
        # ------------------------------------------------------
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device, weights_only=True)
        load_key, no_load_key, temp_dict = [], [], {}

        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print(f"\n[Info] 成功加载 Backbone 权重: {len(load_key)} keys.")
            print(f"[Info] 跳过不匹配权重 (Head部分): {len(no_load_key)} keys.")
            print(f"\033[1;33;44m[Experiment] 当前运行实验: {att_type} (Backbone: {phi})\033[0m")
            print(f"\033[1;32m[Log] 结果将保存在: {save_dir}\033[0m\n")

    # ------------------------------------------------------------------
    #   Loss 记录器
    # ------------------------------------------------------------------
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------
    #   混合精度 Scaler
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    #   读取数据集
    # ------------------------------------------------------------------
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # ------------------------------------------------------------------
    #   打印配置信息
    # ------------------------------------------------------------------
    if local_rank == 0:
        show_config(
            num_classes=num_classes, phi=phi, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        print(f"Hybrid Loss: Dice={dice_loss}, Focal={focal_loss}, Cls_Weights={cls_weights}")

    # ------------------------------------------------------------------
    #   开始训练循环
    # ------------------------------------------------------------------
    if True:
        UnFreeze_flag = False

        # 冻结 Backbone
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 自适应学习率
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 构建 DataLoader
        train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=seg_dataset_collate,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=seg_dataset_collate,
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # ------------------------------------
        # Epoch 循环
        # ------------------------------------
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 解冻阶段逻辑
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # 重新计算学习率
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=seg_dataset_collate,
                                 worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=seg_dataset_collate,
                                     worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集太小，无法训练")

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, UnFreeze_Epoch, Cuda, \
                          dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir,
                          local_rank)

        if local_rank == 0:
            loss_history.writer.close()