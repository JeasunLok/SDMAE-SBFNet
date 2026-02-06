import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os
import torch
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from dataloader import *
from model import *
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.cuda.amp import GradScaler
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 

#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, e, epoch, device, num_classes, scaler, fp16, ignore_index=None):
    loss_show = AverageMeter()
    local_rank = int(os.environ["LOCAL_RANK"])
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    if local_rank == 0:
        loop = tqdm(enumerate(train_loader), total = len(train_loader))
    else:
        loop = enumerate(train_loader)
    for batch_idx, (batch_data, batch_data_hsv, batch_label) in loop:
        batch_data = batch_data.to(device).float()
        batch_data_hsv = batch_data_hsv.to(device).float()
        batch_label = batch_label.to(device).long()

        optimizer.zero_grad()

        if fp16:
            with torch.cuda.amp.autocast():
                batch_prediction, out, embedding = model(batch_data, batch_data_hsv)
                # with torch.cuda.amp.autocast(enabled=False):
                #     loss = criterion(batch_prediction.float(), batch_label)  # 确保损失计算用FP32
                loss = criterion(batch_prediction, batch_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_prediction, out, embedding = model(batch_data, batch_data_hsv)
            loss = criterion(batch_prediction, batch_label)
            loss.backward()
            optimizer.step()     

        batch_prediction = F.softmax(batch_prediction, dim=1)
        batch_prediction = torch.argmax(batch_prediction, dim=1)
        # calculate the accuracy

        CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
        CM = CM + CM_batch
        n = batch_data.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc = compute_acc(CM, ignore_index=ignore_index)
        mIoU = compute_mIoU(CM, ignore_index=ignore_index)

        if local_rank == 0:
            loop.set_description(f'Train Epoch [{e+1}/{epoch}]')
            loop.set_postfix({"train_loss":loss_show.average.item(),
                            "train_accuracy": str(round(acc*100, 2)) + "%",
                            "train_mIoU": str(round(mIoU*100, 2)) + "%"})

    return acc, mIoU, loss_show.average.item()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# valid model
def valid_epoch(model, val_loader, criterion, e, epoch, device, num_classes, ignore_index=None):
    loss_show = AverageMeter()
    local_rank = int(os.environ["LOCAL_RANK"])
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    if local_rank == 0:
        loop = tqdm(enumerate(val_loader), total = len(val_loader))
    else:
        loop = enumerate(val_loader)
    with torch.no_grad():
        for batch_idx, (batch_data, batch_data_hsv, batch_label) in loop:
            batch_data = batch_data.to(device).float()
            batch_data_hsv = batch_data_hsv.to(device).float()
            batch_label = batch_label.to(device).long()

            batch_prediction, out, embedding = model(batch_data, batch_data_hsv)
            loss = criterion(batch_prediction, batch_label)

            batch_prediction = F.softmax(batch_prediction, dim=1)
            batch_prediction = torch.argmax(batch_prediction, dim=1)

            # calculate the accuracy
            CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
            CM = CM + CM_batch
            n = batch_data.shape[0]

            # update the loss and the accuracy 
            loss_show.update(loss.data, n)
            acc = compute_acc(CM, ignore_index=ignore_index)
            mIoU = compute_mIoU(CM, ignore_index=ignore_index)

            if local_rank == 0:
                loop.set_description(f'Val Epoch [{e+1}/{epoch}]')
                loop.set_postfix({"val_loss":loss_show.average.item(),
                                "val_accuracy": str(round(acc*100, 2)) + "%",
                                "val_mIoU": str(round(mIoU*100, 2)) + "%"})

    return CM, acc, mIoU, loss_show.average.item()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# test model
def test_epoch(model, test_loader, device, num_classes, ignore_index=None):
    local_rank = int(os.environ["LOCAL_RANK"])
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    if local_rank == 0:
        loop = tqdm(enumerate(test_loader), total = len(test_loader))
    else:
        loop = enumerate(test_loader)
    with torch.no_grad():
        for batch_idx, (batch_data, batch_data_hsv, batch_label) in loop:
            batch_data = batch_data.to(device).float()
            batch_data_hsv = batch_data_hsv.to(device).float()
            batch_label = batch_label.to(device).long()

            batch_prediction, out, embedding = model(batch_data, batch_data_hsv)

            batch_prediction = F.softmax(batch_prediction, dim=1)
            batch_prediction = torch.argmax(batch_prediction, dim=1)

            # calculate the accuracy
            CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
            CM = CM + CM_batch

            # update the loss and the accuracy 
            acc = compute_acc(CM, ignore_index=ignore_index)
            mIoU = compute_mIoU(CM, ignore_index=ignore_index)

            if local_rank == 0:
                loop.set_description(f'Test Epoch')
                loop.set_postfix({"test_accuracy": str(round(acc*100, 2)) + "%",
                                "test_mIoU": str(round(mIoU*100, 2)) + "%"})

    return CM, acc, mIoU
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    # 是否使用GPU
    Cuda = True
    num_workers = 4
    distributed = True
    sync_bn = True
    fp16 = True
    test = False
    
    num_classes = 8 # LoveDA:6 DLRSD:17 WHDLD:6 PRDLC:8

    model_pretrained = False
    model_path = r""
    encoder_pretrained = True
    encoder_path = r"checkpoints/PRD262K/sdmae/vit-b-sdmae-100.pt"
    freeze_encoder = False

    input_shape = [512, 512]
    epoch = 50
    save_period = 2
    batch_size = 8
    ignore_index = 0

    if ignore_index == 0:
        num_classes = num_classes + 1

    # 学习率
    lr = 1e-4
    min_lr = lr*0.01

    # 优化器
    momentum = 0.9 
    weight_decay = 0
    use_focal_loss = True
    
    data_dir = r"data/segmentation"
    logs_dir = r"logs/segmentation"
    checkpoints_dir = r"checkpoints/segmentation"
    time_now = time.localtime()
    logs_folder = os.path.join(logs_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    checkpoints_folder = os.path.join(checkpoints_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        os.makedirs(logs_folder)
        os.makedirs(checkpoints_folder)

    
    if local_rank == 0:
        print("===============================================================================")
    # 设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if Cuda:
        if distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            init_ddp(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_rank = 0
    else:
        device = torch.device("cpu")
        local_rank = 0

    model = SDMAE_ViT(image_size=input_shape[0], patch_size=32)
    
    if encoder_pretrained:
        if local_rank == 0:
            print('Load weights {}.'.format(encoder_path))
        ddp_model = torch.load(encoder_path, map_location='cuda')
        state_dict = ddp_model.state_dict()
        encoder_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "encoder" in k}
        model.load_state_dict(encoder_state_dict, strict=False)

    model = SDSBFNet_2(model.encoder, out_channels=num_classes, downsample_factor=16)
    model_train = model.train()
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model_train)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # 混精度
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        if local_rank == 0:
            print("Sync_bn is not support in one gpu or not distributed.")

    if model_pretrained and model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_train.load_state_dict(torch.load(model_path))  
        
    with open(os.path.join(data_dir, r"list/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(data_dir, r"list/val.txt"),"r") as f:
        val_lines = f.readlines()
    with open(os.path.join(data_dir, r"list/test.txt"),"r") as f:
        test_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    num_test = len(test_lines)

    torch.manual_seed(4326)
    np.random.seed(4326)

    if local_rank == 0:
        print("device:", device, "num_train:", num_train, "num_val:", num_val, "num_test:", num_test)
        print("===============================================================================")

    # 优化器
    if freeze_encoder:
        for param in [model_train.cls_token, model_train.pos_embedding, model_train.patchify, model_train.transformer, model_train.layer_norm]:
            param.requires_grad = False
        parameters_to_optimize = [
            {'params': [model_train.alpha, model_train.beta]},  # 将 alpha 和 beta 参数添加到优化器中
            {'params': model_train.mask_conv_encoder.parameters()},  # 将 mask_conv_encoder 的参数添加到优化器中
            {'params': model_train.decoder_1.parameters()},  # 将 decoder_1 的参数添加到优化器中
            {'params': model_train.decoder_2.parameters()},  # 将 decoder_2 的参数添加到优化器中
            {'params': model_train.decoder_3.parameters()},  # 将 decoder_3 的参数添加到优化器中
            {'params': model_train.decoder_4.parameters()},  # 将 decoder_4 的参数添加到优化器中
            {'params': model_train.decoder_5.parameters()},  # 将 decoder_5 的参数添加到优化器中
            {'params': model_train.scene_1.parameters()},  # 将 scene_1 的参数添加到优化器中
            {'params': model_train.scene_2.parameters()},  # 将 scene_2 的参数添加到优化器中
            {'params': model_train.scene_3.parameters()},  # 将 scene_3 的参数添加到优化器中
            {'params': model_train.scene_4.parameters()},  # 将 scene_4 的参数添加到优化器中
            {'params': model_train.scene_5.parameters()},  # 将 scene_5 的参数添加到优化器中
            {'params': model_train.origin_scene_embedding.parameters()},  # 将 origin_scene_embedding 的参数添加到优化器中
            {'params': model_train.final_out.parameters()},  # 将 final_out 的参数添加到优化器中
        ]
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model_train.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch//10, eta_min=min_lr)

    if use_focal_loss:
        criterion = FocalLoss(ignore_index=ignore_index)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()

    image_transform = get_transform(input_shape, IsResize=True, IsTotensor=True, IsNormalize=True)
    label_transform = get_transform(input_shape, IsResize=True, IsTotensor=False, IsNormalize=False)
    train_dataset = SDSegDataset(train_lines, input_shape, num_classes, image_transform=image_transform, label_transform=label_transform)
    val_dataset = SDSegDataset(val_lines, input_shape, num_classes, image_transform=image_transform, label_transform=label_transform)
    test_dataset = SDSegDataset(test_lines, input_shape, num_classes, image_transform=image_transform, label_transform=label_transform)

    if distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        test_sampler     = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False,)
        batch_size      = batch_size // ngpus_per_node
        shuffle         = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=test_sampler)

    # 开始模型训练
    if not test:
        if local_rank == 0:
            print("start training")
        epoch_result = np.zeros([4, epoch])
        best_loss = 1e9
        for e in range(epoch):
            model_train.train()
            train_acc, train_mIoU, train_loss = train_epoch(model_train, train_loader, criterion, optimizer, e, epoch, device, num_classes, scaler, fp16, ignore_index=ignore_index)
            scheduler.step()
            if local_rank == 0:
                print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.2f}% | train_mIoU: {:.2f}%".format(e+1, train_loss, train_acc*100, train_mIoU*100))
            epoch_result[0][e], epoch_result[1][e], epoch_result[2][e], epoch_result[3][e]= e+1, train_loss, train_acc*100, train_mIoU*100

            if ((e+1) % save_period == 0) or (e == epoch - 1):
                if local_rank == 0:
                    print("===============================================================================")
                    print("start validating")
                model_train.eval()      
                val_CM, val_acc, val_mIoU, val_loss = valid_epoch(model_train, val_loader, criterion, e, epoch, device, num_classes, ignore_index=ignore_index)
                val_weighted_recall, val_weighted_precision, val_weighted_f1, IoU_array, Precision_array, Recall_array, F1_array = compute_metrics(val_CM, ignore_index=ignore_index)
                if local_rank == 0:
                    print("Epoch: {:03d}  =>  Accuracy: {:.2f}% | MIoU: {:.2f}% | W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(e+1, val_acc*100, val_mIoU*100, val_weighted_recall, val_weighted_precision, val_weighted_f1))
                    torch.save(model_train, os.path.join(checkpoints_folder, "model_loss" + str(round(val_loss, 4)) + "_epoch" + str(e+1) + ".pt"))
                    torch.save(model_train.state_dict(), os.path.join(checkpoints_folder, "model_state_dict_loss" + str(round(val_loss, 4)) + "_epoch" + str(e+1) + ".pth"))
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model_train, os.path.join(checkpoints_folder, "model_best.pt"))
                        torch.save(model_train.state_dict(), os.path.join(checkpoints_folder, "model_state_dict_best.pth"))
                    print("===============================================================================")
        
        if distributed:
            train_sampler.set_epoch(epoch)

        if distributed:
            dist.barrier()

        if local_rank == 0:
            print("save train logs successfully")
            draw_result_visualization(logs_folder, epoch_result)

    if local_rank == 0:
        print("===============================================================================")
        print("start testing")
    if not test:
        model_train.load_state_dict(torch.load(os.path.join(checkpoints_folder, "model_state_dict_best.pth")))  
    model_train.eval()
    test_CM, test_acc, test_mIoU = test_epoch(model_train, test_loader, device, num_classes, ignore_index=ignore_index)
    test_weighted_recall, test_weighted_precision, test_weighted_f1, IoU_array, Precision_array, Recall_array, F1_array = compute_metrics(test_CM, ignore_index)
    if local_rank == 0:
        print("Test Result  =>  Accuracy: {:.2f}%| mIoU: {:.2f}% | W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(test_acc*100, test_mIoU*100, test_weighted_recall, test_weighted_precision, test_weighted_f1))
        store_result(logs_folder, test_acc, test_mIoU, test_weighted_recall, test_weighted_precision, test_weighted_f1, test_CM, IoU_array, Precision_array, Recall_array, F1_array, epoch, batch_size, lr, weight_decay)
        print("save test result successfully")
        print("===============================================================================") 

# torchrun --nproc_per_node=2 train_SDSBFNet_ablation_FBM.py
