import os
import argparse
import math
import torch
import torch.nn as nn
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from dataloader import *
from model import *
from utils import setup_seed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--input_shape', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_device_batch_size', type=int, default=8)
    parser.add_argument('--base_learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--sdmask_ratio', type=float, default=0.9)
    parser.add_argument('--total_epoch', type=int, default=99)
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--data_list_path', type=str, default='data/PRD289K/list')
    parser.add_argument('--pretrained_model_path', type=str, default='checkpoints/PRD289K/vit-b-sdmae-noe-1-dict.pth')
    parser.add_argument('--save_model_path', type=str, default='checkpoints/PRD289K/vit-b-sdmae-noe.pt')

    args = parser.parse_args()
    setup_seed(args.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    init_ddp(local_rank)
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    device = torch.device("cuda", local_rank)

    time_now = time.localtime()
    logs_folder = os.path.join("logs/PRD289K", time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    if local_rank == 0:
        os.makedirs(logs_folder)
    input_shape = [args.input_shape, args.input_shape]
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    with open(os.path.join(args.data_list_path, "train.txt")) as f:
        train_lines = f.readlines()
    with open(os.path.join(args.data_list_path, "val.txt")) as f:
        val_lines = f.readlines()
    image_transform = get_transform(input_shape, IsResize=False, IsTotensor=True, IsNormalize=True)
    train_dataset = SDDataset(train_lines, input_shape, image_transform=image_transform)
    val_dataset = SDDataset(val_lines, input_shape, image_transform=image_transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=True)
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(logs_folder, 'SummaryWriter'))

    model = SDMAE_ViT(mask_ratio=args.mask_ratio, sdmask_ratio=args.sdmask_ratio).to(device)
    # model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # BN层同步
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if local_rank == 0:
            print('use {} gpus!'.format(num_gpus))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if args.pretrained and args.pretrained_model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(args.pretrained_model_path))
        model.load_state_dict(torch.load(args.pretrained_model_path))  
    
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=False) # type: ignore

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        if local_rank == 0:
            tqdm_loader = tqdm(iter(dataloader))
        else:
            tqdm_loader = iter(dataloader)

        for img, v_img, e_img in tqdm_loader:
            step_count += 1
            img = img.to(device)
            v_img = v_img.to(device)
            # e_img = e_img.to(device)
            predicted_img, mask = model(img, v_img)
            # predicted_img, mask = model(img)
            loss1 = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            # loss2 = torch.mean((predicted_img - img) ** 2 * mask * e_img) / (torch.sum(mask * e_img)/torch.flatten(mask * e_img).shape[0])
            # loss = loss1 + loss2
            loss = loss1
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        if local_rank == 0:
            writer.add_scalar('sdmae_loss', avg_loss, global_step=e)
            print(f'In epoch {e+1}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        if dist.get_rank() == 0:
            model.eval()
            with torch.no_grad():
                val_img = torch.stack([val_dataset[i][0] for i in range(16)])
                val_v_img = torch.stack([val_dataset[i][1] for i in range(16)])
                val_img = val_img.to(device)
                val_v_img = val_v_img.to(device)
                predicted_val_img, mask = model(val_img, val_v_img)
                # predicted_val_img, mask = model(val_img)

                predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
                img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
                img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
                if local_rank == 0:
                    writer.add_image('mae_image', (img + 1) / 2, global_step=e)
            
            ''' save model '''
            torch.save(model, args.save_model_path.split(".")[0]+"-"+str(e+1+1)+".pt")
            torch.save(model.state_dict(), args.save_model_path.split(".")[0]+"-"+str(e+1+1)+"-dict.pth")

    dist.destroy_process_group()  # 消除进程组，和 init_process_group 相对

# torchrun --nproc_per_node=4 sdmae_pretrain_ddp.py
# tensorboard --logdir=/home/ljs/PRD-RSMAE/PRD-RSMAE/logs/PRD289K/2024-06-13-20-49-07_SDMAE_noe/SummaryWriter --port=6063
# ssh -NfL 8084:127.0.0.1:6063 ljs@172.18.206.54 -p 6522
