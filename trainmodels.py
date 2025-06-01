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
# ==================== 1. 参数配置 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='Super resolution for Hyperspectral images')
    # 数据集参数
    parser.add_argument('--dataset_name', type=str, default='houston',choices=['botswana','pavia','pavia_U','houston'],
                        help='数据集选择')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集路径')
    parser.add_argument('--patch_size', type=int, default=64, help='训练数据集图像大小')
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
    parser.add_argument('--SS2D_MBs_num', type=int, default=0, help='spafem中SS2D_MB数量')
    parser.add_argument('--SS1D_MBs_num', type=int, default=0, help='spefem中SS1D_MB数量')
    parser.add_argument('--SS3D_MBs_num', type=int, default=0, help='_3Dlpfem中SS3D_MB数量')
    parser.add_argument('--spafem_N1', type=int, default=2, help='spafem数量')
    parser.add_argument('--spefem_N2', type=int, default=2, help='spefem数量')
    parser.add_argument('--DFEM_block_N3', type=int, default=1, help='dfem_block数量')
    parser.add_argument('--upscale_factor', type=int, default=2, choices=[2, 4, 8], help='超分上采样系数')

    
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


    train_set = MatImageset( train_data, patch_size=args.patch_size, stride=10, upscale_factor=args.upscale_factor, )
    test_set = MatImageset( test_data, patch_size=args.patch_size, stride=10, upscale_factor=args.upscale_factor, )
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


# ==================== 5. 训练函数 ====================
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,args):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    for batch_idx,(input_lr, upscaled_data, target_hr) in enumerate(pbar):
        if(args.model_name=='d3fcn'):
            input_lr=upscaled_data.unsqueeze(1)
        input_lr,input_hr,target_hr = input_lr.to(device), upscaled_data.to(device), target_hr.to(device)
        input_lr = input_lr.float()
        target_hr = target_hr.float()
        input_hr=input_hr.float()
        # 前向传播
        if(args.model_name=='sspsr'):
            outputs_hr = model(input_lr,input_hr)
        elif(args.model_name=='gdrrn'):
            outputs_hr = model(input_hr)
        elif(args.model_name=='d3fcn'):
            outputs_hr = model(input_lr)
            outputs_hr=outputs_hr.squeeze(1)
        elif(args.model_name=='mymodel'or args.model_name=='essaformer'):
            outputs_hr = model(input_lr)
        loss = criterion(outputs_hr,target_hr)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        
        pbar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}")
    
    epoch_loss = running_loss / (batch_idx+1)
    
    return epoch_loss


# ==================== 6. 验证函数 ====================
@torch.no_grad()
def validate(model, val_loader, criterion, device,args):
    model.eval()
    running_loss = 0.0
    avg_psnr = 0
    avg_sam = 0
    avg_ssim = 0
    avg_ergas = 0
    #eval_f=HSREVAL_METRICS(args.upscale_factor,device=device)
    with torch.no_grad():
        pbar=tqdm(val_loader, desc='Validation')
        for batch_idx,(input_lr, upscaled_data, target_hr) in enumerate(pbar):
            if(args.model_name=='d3fcn'):
                input_lr=upscaled_data.unsqueeze(1)
            input_lr,input_hr,target_hr = input_lr.to(device), upscaled_data.to(device), target_hr.to(device)
            input_lr = input_lr.float()
            target_hr = target_hr.float()
            input_hr=input_hr.float()
            
            if(args.model_name=='sspsr'):
                outputs_hr = model(input_lr,input_hr)
            elif(args.model_name=='gdrrn'):
                outputs_hr = model(input_hr)
            elif(args.model_name=='d3fcn'):
                outputs_hr = model(input_lr)
                outputs_hr=outputs_hr.squeeze(1)
            else:
                outputs_hr = model(input_lr)
            loss = criterion(outputs_hr,target_hr)
            
            #outputs_hr = outputs_hr.unsqueeze(1)#因为评估指标要求输入的形状(B, C, D, H, W)
            #target_hr = target_hr.unsqueeze(1)
            #outputs_hr = outputs_hr.data.cpu().numpy()
            outputs_hr[outputs_hr < 0] = 0
            running_loss += loss.item()
            #psnr,ssim,sam,ergas=eval_f(outputs_hr,target_hr)
            psnr=cal_psnr(target_hr,outputs_hr)
            sam=cal_sam(target_hr,outputs_hr)
            ssim=cal_ssim(target_hr,outputs_hr)
            ergas=cal_ergas(target_hr,outputs_hr,args.upscale_factor)
            #avg_psnr += np.mean(OUTmpsnr)
            avg_psnr += psnr
            avg_sam += sam
            avg_ssim += ssim
            avg_ergas += ergas
            pbar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}",avSpa_PSNR=f"{avg_psnr/(batch_idx+1):.4f}",Spa_SAM=f"{avg_sam/(batch_idx+1):.4f}",Spa_SSIM=f"{avg_ssim/(batch_idx+1):.4f}",Spa_ERGAS=f"{avg_ergas/(batch_idx+1):.4f}")
    
    # 计算各项指标
    epoch_loss = running_loss /(batch_idx+1)
    psnr = avg_psnr/(batch_idx+1)
    sam = avg_sam/(batch_idx+1)
    ssim =avg_ssim/(batch_idx+1)
    ergas=avg_ergas/(batch_idx+1)
    
    return epoch_loss,psnr,sam,ssim,ergas
# ==================== 7. 保存检查点函数 ====================
def checkpoint(epoch,args,trained_model,upscale_rate=4):
    if args.upscale_factor == upscale_rate and args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        # print('saving')
        torch.save(trained_model.state_dict(), os.path.join(args.save_dir, f'{args.dataset_name}/{args.model_name}_epoch{epoch}_x{upscale_rate}.pth'))
## ==================== 7. 打印网络 ====================
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
# ==================== 7. 主函数 ====================
def main():
    args = parse_args()
    
    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    torch.manual_seed(args.seed)  # set the seed of the random number generator to a fixed value
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 数据加载
    HSI_Channels,train_loader,val_loader=load_data(args)
    
    
    # 模型构建
    model = build_model(HSI_Channels,args).to(device)
    print('---------- Networks architecture -------------')
    print_network(model)
    # print_network(generator_spe)
    print('----------------------------------------------')
    # 损失函数和优化器True
    criterion = HybridLoss(spatial_tv=True,spectral_tv=True).to(device)
    
    
    # 训练循环
    best_val = -100000
    train_history = {'loss': []}
    val_history = {'loss': [], 'PSNR': [], 'SAM': [], 'SSIM': [], 'ERGAS': []}
    start = time.time()
    if(args.model_name=='bicubic'):
        val_loss, val_psnr, val_sam, val_ssim, val_ergas = validate(model, val_loader, criterion, device,args)
        val_history['loss'].append(val_loss)
        val_history['PSNR'].append(val_psnr)
        val_history['SAM'].append(val_sam)
        val_history['SSIM'].append(val_ssim)
        val_history['ERGAS'].append(val_ergas)
    
        print(f'Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, '
              f'SAM: {val_sam:.4f}, SSIM: {val_ssim:.4f}, ERGAS: {val_ergas:.4f}')
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1,args.b2),weight_decay=args.weight_decay,amsgrad=args.amsgrad)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=args.gamma)
        for epoch in range(args.epoch, args.n_epochs):
            print("===> Start training the {} epoch. ".format(epoch))
            # 训练
            train_loss= train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,args)
            train_history['loss'].append(train_loss)
            
            # 验证
            val_loss, val_psnr, val_sam, val_ssim, val_ergas = validate(model, val_loader, criterion, device,args)
            val_history['loss'].append(val_loss)
            val_history['PSNR'].append(val_psnr)
            val_history['SAM'].append(val_sam)
            val_history['SSIM'].append(val_ssim)
            val_history['ERGAS'].append(val_ergas)
            
            # 学习率调整
            #scheduler.step()
            
            # 保存模型
            if val_psnr> best_val:
                best_val = val_psnr
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.dataset_name}/best_{args.model_name}_x{args.upscale_factor}.pth'))
            
            checkpoint(epoch,args,model,args.upscale_factor)
            
            # 打印信息
            print(f'Epoch {epoch+1}/{args.n_epochs}:')
            print(f'Train - Loss: {train_loss:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, '
                f'SAM: {val_sam:.4f}, SSIM: {val_ssim:.4f}, ERGAS: {val_ergas:.4f}')
        end = time.time()
        print("===> Total time is: {:.2f}".format(end - start))
        # 保存训练历史
        torch.save({
            'train_history': train_history,
            'val_history': val_history,
            'args': vars(args)
        }, os.path.join(args.save_dir, f'{args.dataset_name}/{args.model_name}_training_history_x{args.upscale_factor}.pt'))
    
    


if __name__ == '__main__':
    main()