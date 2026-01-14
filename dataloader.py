import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from utils import *

class MyDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, image_transform=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.image_transform = image_transform

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name_image = annotation_line.split()[0]
        image = Image.open(name_image)

        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))
        return image
    
    def __len__(self):
        return self.length

class SDDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, image_transform=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.image_transform = image_transform

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name_image = annotation_line.split()[0]
        image = Image.open(name_image)

        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 1 ,0]))
        
        img_np = np.array(image.squeeze(0))
        img_hsv = torch.from_numpy(rgb_to_hsv(img_np)[:, :, 2]).unsqueeze(0)
        img_edge = torch.from_numpy(detect_edges_with_laplacian_and_otsu(img_np)).unsqueeze(0)
        img_edge = torch.cat([img_edge] * 3, dim=0).byte()

        return image, img_hsv, img_edge
    
    def __len__(self):
        return self.length
    
class SegDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, image_transform=None, label_transform=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name_label = annotation_line.split()[0]
        # name_image = name_label.replace("label", "image").split(".")[0] + ".png" # LoveDA
        # name_image = name_label.replace("label", "image").split(".")[0] + ".tif" # DLRSD
        name_image = name_label.replace("label", "image").split(".")[0] + ".jpg" # WHDLD PRDLC

        image = Image.open(name_image)
        label = Image.open(name_label)

        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))
        if self.label_transform is not None:
            label = self.label_transform(label)
            label = torch.tensor(np.array(label))
        else:
            label= torch.from_numpy(np.array(label))
        return image, label

    def __len__(self):
        return self.length

class SDSegDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, image_transform=None, label_transform=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name_label = annotation_line.split()[0]
        # name_image = name_label.replace("label", "image").split(".")[0] + ".png" # LoveDA
        # name_image = name_label.replace("label", "image").split(".")[0] + ".tif" # DLRSD
        name_image = name_label.replace("label", "image").split(".")[0] + ".jpg" # WHDLD PRDLC

        image = Image.open(name_image)
        label = Image.open(name_label)

        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))
        if self.label_transform is not None:
            label = self.label_transform(label)
            label = torch.tensor(np.array(label))
        else:
            label= torch.from_numpy(np.array(label))

        img_np = np.array(image.squeeze(0))
        img_hsv = torch.from_numpy(rgb_to_hsv(img_np)[:, :, 2]).unsqueeze(0)

        return image, img_hsv, label

    def __len__(self):
        return self.length
    

if __name__ == "__main__":
    annotation_lines = [r"E:\202403\PRD-RSMAE\PRD-RSMAE\F49E001015_Level_17_1009.TIF", r"E:\202403\PRD-RSMAE\PRD-RSMAE\F49E001015_Level_17_1009.TIF"]  # 你需要替换为实际的文件路径  
    input_shape = [512, 512]  # 假设的输入形状，你可能需要根据你的模型进行调整  
        
    # 创建一个SDDataset实例  
    dataset = SDDataset(annotation_lines, input_shape)  
        
    # 创建一个DataLoader来加载数据  
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  
    # 遍历DataLoader并打印结果  
    for image, img_hsv, img_edge in data_loader:  
        print(f"Image shape: {image.shape}")  
        print(f"HSV shape: {img_hsv.shape}")  
        print(f"Edge shape: {img_edge.shape}")
        a = img_edge
        b = image

    plt.figure()
    b = np.transpose(np.array(b.squeeze(0)))*255
    print(b.shape)
    plt.imshow(b, cmap='gray')
    plt.show()
    
    plt.figure()
    a = np.transpose(np.array(a.squeeze(0)))*255
    print(a.shape)
    plt.imshow(a, cmap='gray')
    plt.show()
    