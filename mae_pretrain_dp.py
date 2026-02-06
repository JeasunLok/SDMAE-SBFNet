import os
import argparse
import math
import torch
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from dataloader import *
from models import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--input_shape', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_device_batch_size', type=int, default=16)
    parser.add_argument('--base_learning_rate', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--warmup_epoch', type=int, default=20)
    parser.add_argument('--data_list_path', type=str, default='data/PRD289K/list')
    parser.add_argument('--pretrained_model_path', type=str, default='logs/PRD289K/vit-b-mae-101-dict.pth')
    parser.add_argument('--save_model_path', type=str, default='logs/PRD289K/vit-b-mae.pt')

    args = parser.parse_args()
    setup_seed(args.seed)

    time_now = time.localtime()
    logs_folder = os.path.join("logs/PRD289K", time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
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
    train_dataset = MyDataset(train_lines, input_shape, image_transform=image_transform)
    val_dataset = MyDataset(val_lines, input_shape, image_transform=image_transform)

    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join(logs_folder, 'SummaryWriter'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    model = torch.nn.DataParallel(model)

    if args.pretrained and args.pretrained_model_path != '':
        print('Load weights {}.'.format(args.pretrained_model_path))
        model.load_state_dict(torch.load(args.pretrained_model_path))  
    
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        torch.save(model, args.save_model_path.split(".")[0]+"-"+str(e+1)+".pt")
        torch.save(model.state_dict(), args.save_model_path.split(".")[0]+"-"+str(e+1)+"-dict.pth")

# python mae_pretrain_dp.py
# tensorboard --logdir=/home/ljs/PRD-RSMAE/PRD-RSMAE/logs/PRD289K/2024-03-14-18-43-56/SummaryWriter --port=6062
# ssh -NfL 10052:127.0.0.1:6062 ljs@172.18.206.54 -p 6522
