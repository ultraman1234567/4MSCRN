import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import bisect
import numpy as np
from data_preprocess.gaussian_down_sample import gaussian_down_sample
import torchvision.transforms as transforms
import torch
import cv2
from scipy.ndimage import gaussian_filter
import math

def load_data(dataset_name):
    if dataset_name == 'botswana':
        file_name = './data/Botswana.mat'
        data = sio.loadmat(file_name)['Botswana']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
    elif dataset_name == 'pavia':
        file_name = './data/Pavia.mat'
        data = sio.loadmat(file_name)['pavia']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
    elif dataset_name == 'pavia_U':
        file_name = './data/PaviaU.mat'
        data = sio.loadmat(file_name)['paviaU']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
    elif dataset_name == 'houston':
        file_name = './data/Houston.mat'
        data = sio.loadmat(file_name)['Houston']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    return data
def showdata(data,channels):
    data = data[channels, :, :].transpose(1, 2, 0)  # 调整维度顺序为 (H, W, C)
    
    # 归一化数据
    data = (data - data.min()) / (data.max() - data.min())
    
    # 绘制图像
    plt.imshow(data)
    #plt.title("Random Channels Visualization")
    plt.show()
    

if __name__ == '__main__':
    # 加载数据
        data = load_data('houston')
        # print(index)
        # input_lr = gaussian_down_sample(self.data_tensor[:, h:h + patch_size, w:w + patch_size], self.upscale_factor)
        # input_lr = self.data_tensor[:, h:h + patch_size, w:w + patch_size]
        D, H, W=data.shape
        h=250
        w=120
        input_lr = data[:, h:h + 64, w:w + 64]
        target_hr = data[:, h:h + 64, w:w + 64]
        upscale_factor=4
        filtered_data = np.zeros_like(input_lr, dtype=np.float32)
        for i in range(input_lr.shape[0]):
            filtered_data[i, :, :] = gaussian_filter(input_lr[i, :, :], sigma=math.sqrt(0.72), radius=2)

        # 平均池化下采样
        downsampled_data = np.zeros(
            (input_lr.shape[0], input_lr.shape[1] // upscale_factor, input_lr.shape[2] // upscale_factor))
        for i in range(input_lr.shape[0]):
            downsampled_data[i, :, :] = cv2.resize(filtered_data[i, :, :],
                                                   (downsampled_data.shape[2], downsampled_data.shape[1]),
                                                   interpolation=cv2.INTER_CUBIC)
        upscaled_data = np.zeros_like(filtered_data)
        for i in range(filtered_data.shape[0]):
            upscaled_data[i, :, :] = cv2.resize(downsampled_data[i, :, :],
                                                (filtered_data.shape[2], filtered_data.shape[1]),
                                                interpolation=cv2.INTER_CUBIC)
        # for interpolation type SR, comment the next line of code
        # input_lr = gaussian_down_sample(input_lr, self.upscale_factor)
        # target_hr = self.target_tensor[:, h:h+patch_size, w:w+patch_size]
        # 归一化到 [0, 1]（避免显示异常）
        #downsampled_data = (downsampled_data - downsampled_data.min()) / (downsampled_data.max() - downsampled_data.min())
        #upscaled_data = (upscaled_data - upscaled_data.min()) / (upscaled_data.max() - upscaled_data.min())
        #target_hr = (target_hr - target_hr.min()) / (target_hr.max() - target_hr.min())
        showdata(upscaled_data,list([0,30,90]))