import torch
from fvcore.nn import FlopCountAnalysis
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
from DFEM import DFEM,SpaFEM,SpeFEM,_3DLPFEM
from SFEM import *
from mymodel import mymodel,xxx,bicubic_up,xx
from dataset.dataset_iter import MatImageset
from eval.accuracy_cuda import *
from loss.hybrid_loss import HybridLoss
from loss.sam_loss import SamLoss
from othermodels.d3fcn import d3fcn
from othermodels.sspsr import SSPSR,default_conv
from othermodels.gdrrn import GDRRN
from othermodels.essaformer import ESSA
from matplotlib.lines import Line2D
def parse_args():
    parser = argparse.ArgumentParser(description='Super resolution for Hyperspectral images')
    # 数据集参数
    parser.add_argument('--dataset_name', type=str, default='pavia_U',choices=['botswana','pavia','pavia_U','houston'],
                        help='数据集选择')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集路径')
    parser.add_argument('--patch_size', type=int, default=64*4, help='训练数据集图像大小')
    # 训练参数
    
    parser.add_argument('--epoch', type=int, default=0, help='训练开始的轮数')
    parser.add_argument('--n_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batchSize', type=int, default=32, help='训练批次大小')
    parser.add_argument('--testBatchSize', type=int, default=32, help='推理批次大小')
    parser.add_argument('--lr', type=float, default=0.002, help='adam: 初始学习率')
    
    parser.add_argument('--b1', type=float, default=0.9, help='动量衰减系数（β₁ 控制一阶矩，β₂ 控制二阶矩）'
                        'coefficients used for computing running averages of gradient and its square')
    parser.add_argument('--b2', type=float, default=0.999, help='动量衰减系数（β₁ 控制一阶矩，β₂ 控制二阶矩）')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2正则化-权重衰减')
    parser.add_argument('--amsgrad', action='store_true', default=False, help='是否使用 AMSGrad 变体')
    parser.add_argument('--decay_epoch', type=int, default=40, help='学习率衰减的步长（单位：epoch）')
    parser.add_argument('--gamma', type=float, default=0.5, help='衰减系数（新学习率 = 旧学习率 * gamma）')
    
    # 设备参数
    parser.add_argument('--n_cpu', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpu', type=str, default='0', help='the number of gpu id, only one number')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='mymodel', 
                       choices=['mymodel','d3fcn', 'sspsr', 'gdrrn','essaformer','bicubic'], help='模型名称')
    parser.add_argument('--pretrained', action='store_true', default=False,help='使用预训练模型')
    parser.add_argument('--hidden_channels', type=int, default=64, help='深度特征提取的光谱通道数')
    parser.add_argument('--SS2D_MBs_num', type=int, default=5, help='spafem中SS2D_MB数量')
    parser.add_argument('--SS1D_MBs_num', type=int, default=8, help='spefem中SS1D_MB数量')
    parser.add_argument('--SS3D_MBs_num', type=int, default=2, help='_3Dlpfem中SS3D_MB数量')
    parser.add_argument('--spafem_N1', type=int, default=4, help='spafem数量')
    parser.add_argument('--spefem_N2', type=int, default=4, help='spefem数量')
    parser.add_argument('--DFEM_block_N3', type=int, default=2, help='dfem_block数量')
    parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8], help='超分上采样系数')

    
    # 保存参数
    parser.add_argument('--checkpoint_interval', type=int, default=-1, help='保存间隔(epoch)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
    #-------------------------------
  

    return parser.parse_args()


# ==================== 2. 自定义数据集 ====================



# ==================== 3. 数据预处理 ====================

# ==================== 3. 数据加载：返回数据加载器 ====================
def load_data(args):
    '''
    数据加载器是迭代器，每次迭代产生的数据形状(B,C,H,W)
    '''
    HSI_Channels=0
    if(args.dataset_name=='botswana'):
        
        file_name = './data/Botswana.mat'
        data = sio.loadmat(file_name)['Botswana']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
        train_data = data[:, :, :]
        test_data = data[:, 1220:, :]
        HSI_Channels=145
    elif(args.dataset_name=='pavia'):
        file_name = './data/Pavia.mat'
        data = sio.loadmat(file_name)['pavia']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
        train_data = data[:, :, :]
        test_data = data[:, 946:, :150]
        HSI_Channels=102
    elif(args.dataset_name=='pavia_U'):
        file_name = './data/PaviaU.mat'
        data = sio.loadmat(file_name)['paviaU']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
        train_data = data[:, :, :]
        test_data = data[:, 420:, :140]
        HSI_Channels=103
    elif(args.dataset_name=='houston'):
        file_name = './data/Houston.mat'
        data = sio.loadmat(file_name)['Houston']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
        train_data = data[:, :, :]
        test_data = data[:, 230:, :150]
        HSI_Channels=144
    # file_name2 = '../data/Pavia_bicubic_x2.mat'
    # data2 = sio.loadmat(file_name2)['Pavia_bicubic_x2']
    #
    # # file_name2 = './data/Pavia_bicubic_x3.mat'
    # # data2 = sio.loadmat(file_name2)['Pavia_bicubic_x3']
    #
    #file_name2 = 'data/Pavia.mat'
    #data2 = sio.loadmat(file_name2)['Pavia_bicubic_x4']

    #data2 = np.transpose(data2, [2, 0, 1]).astype(np.float32)
    #train_data2 = data2[:, :946, :]
    #test_data2 = data2[:, 946:, :150]


    train_set = MatImageset( train_data, patch_size=args.patch_size, stride=16, upscale_factor=args.upscale_factor, )
    test_set = MatImageset( test_data, patch_size=args.patch_size, stride=16, upscale_factor=args.upscale_factor, )
    train_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True, num_workers=args.n_cpu, pin_memory=True)
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False, num_workers=args.n_cpu, pin_memory=True)
    return HSI_Channels,train_data_loader, test_data_loader


# ==================== 4. 模型选择定义 ====================
def build_model(HSI_Channels,args):
    if args.model_name=='mymodel':
        #sfem=xx(HSI_Channels,args.hidden_channels).to('cuda')
        sfem=SFEM(HSI_Channels,args.hidden_channels,0).to('cuda')
        spafem=SpaFEM((args.hidden_channels,int(args.patch_size/args.upscale_factor),int(args.patch_size/args.upscale_factor)),args.hidden_channels,(8,8),64,SS2D_MBs_num=args.SS2D_MBs_num,device='cuda').to('cuda')
        spefem=SpeFEM((args.hidden_channels,int(args.patch_size/args.upscale_factor),int(args.patch_size/args.upscale_factor)),args.hidden_channels,16,SS1D_MBs_num=args.SS1D_MBs_num,device='cuda').to('cuda')
        _3dlpfem=_3DLPFEM((args.hidden_channels,int(args.patch_size/args.upscale_factor),int(args.patch_size/args.upscale_factor)),args.hidden_channels,(16,8,8),32,SS3D_MBs_num=args.SS3D_MBs_num,device='cuda').to('cuda')
        dfem=DFEM(args.spafem_N1,args.spefem_N2,spafem,spefem,args.hidden_channels,_3dlpfem,args.DFEM_block_N3).to('cuda')
        #dfem=xxx()
        model=mymodel(args.hidden_channels,HSI_Channels,sfem,dfem,args.upscale_factor).to('cuda')
        #model.load_state_dict(torch.load(f"./checkpoints/{args.dataset_name}/best_{args.model_name}_x{args.upscale_factor}.pth"))
    elif args.model_name == 'd3fcn':
        model = d3fcn()
    elif args.model_name == 'bicubic':
        model= bicubic_up(args.upscale_factor)
    elif args.model_name == 'sspsr':
        n_subs = 8
        n_ovls = 1
        colors = HSI_Channels
        n_blocks = 4
        n_feats = 256
        n_scale = args.upscale_factor

        model= SSPSR(n_subs=n_subs, n_ovls=n_ovls, n_colors=colors, n_blocks=n_blocks, n_feats=n_feats,
                     
                            n_scale=n_scale, res_scale=0.1, use_share=True, conv=default_conv)
    elif args.model_name =='gdrrn':
        model= GDRRN(input_chnl_hsi=HSI_Channels)
    elif args.model_name =='essaformer':
        model= ESSA(HSI_Channels, dim=256, upscale=args.upscale_factor)
    return model
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(f"Params: {num_params / 1e3} K")  # 以百万为单位
if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 102, 64, 64).to(device)
    lx=torch.randn(1, 102, 64*4, 64*4).to(device)
    xxxx=torch.randn(1,1, 102, 64, 64).to(device)
    model=build_model(102,args).to(device)
    model.eval()
    print_network(model)
    flops = FlopCountAnalysis(model, x).total()
    print(f"FLOPs: {flops / 1e9} G")  # 以十亿次浮点运算为单位
""" if __name__ == '__main__':
 

 

    # 数据
    methods = ['3DFCN', 'GDRRN', 'SSPSR', 'ESSA', 'Ours']
    params = [39, 375, 16025, 11521, 9377]
    FLOPs = [16.42, 122.73, 632.03, 829.98, 12.91]
    psnr = [26.3304, 25.8965, 26.3042, 26.2533, 26.2894]

    # 创建气泡图
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # 为每个气泡设置不同的颜色

    # 绘制气泡图
    for i, method in enumerate(methods):
        plt.scatter(FLOPs[i], psnr[i], s=params[i]*0.1, alpha=0.5, color=colors[i], label=f'{method} ({params[i]}k)')

    # 添加标签
    for i, method in enumerate(methods):
        plt.text(FLOPs[i], psnr[i], method, fontsize=9, ha='right')

    # 设置图表标题和坐标轴标签
    #plt.title('Model Comparison')
    plt.xlabel('FLOPs (G)')
    plt.ylabel('PSNR (dB)')

    # 创建图例句柄列表
    legend_handles = []
    for i, size in enumerate(params):
        handle = Line2D([0], [0], marker='o', color='w', label=f'{methods[i]} ({size}k)', markerfacecolor=colors[i], markersize=10, alpha=0.5)
        legend_handles.append(handle)
    # 添加图例
    plt.legend(handles=legend_handles, title="Methods_Params")
    # 显示图表
    plt.grid(True)
    plt.show() """