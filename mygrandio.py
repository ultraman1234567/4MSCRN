import gradio as gr
import numpy as np
from PIL import Image
import tempfile
import os
import torch
import cv2
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
from scipy.ndimage import gaussian_filter
import math
import random
import tkinter.font as tkFont
import sys
import locale
if sys.platform.startswith('win'):
    locale.setlocale(locale.LC_ALL, 'chinese')
else:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 在import任何Qt相关库之前添加
# 1. 初始化所有超分模型
class HSI_Models:
    def __init__(self,upscale=4,patch_size=224):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {
            "4MSCRN": self.init_4MSCRN,
            "3dfcn": self.init_3dfcn,
            "sspsr": self.init_sspsr,
            "gdrrn": self.init_gdrrn,
            "essaformer":self.init_essaformer
        }
        self.model_instances = {}
        self.scale=upscale
        self.hsi_data=None
        self.selected_channels=None
        self.lr=None
        self.bicubic_hr=None
        self.model_hr=None
        self.patch_size=patch_size
    def init_4MSCRN(self):
        #sfem=xx(HSI_Channels,args.hidden_channels).to('cuda')
        HSI_Channels=102
        hidden_channels=128
        patch_size=self.patch_size
        spafem_N1=1
        spefem_N2=1
        DFEM_block_N3=1
        sfem=SFEM(HSI_Channels,hidden_channels,0).to('cuda')
        spafem=SpaFEM((hidden_channels,int(patch_size/self.scale),int(patch_size/self.scale)),hidden_channels,(8,8),64,SS2D_MBs_num=4,device='cuda').to('cuda')
        spefem=SpeFEM((hidden_channels,int(patch_size/self.scale),int(patch_size/self.scale)),hidden_channels,64,SS1D_MBs_num=8,device='cuda').to('cuda')
        _3dlpfem=_3DLPFEM((hidden_channels,int(patch_size/self.scale),int(patch_size/self.scale)),hidden_channels,(16,8,8),32,SS3D_MBs_num=2,device='cuda').to('cuda')
        dfem=DFEM(spafem_N1,spefem_N2,spafem,spefem,hidden_channels,_3dlpfem,DFEM_block_N3).to('cuda')
        #dfem=xxx()
        model=mymodel(hidden_channels,HSI_Channels,sfem,dfem,self.scale).to('cuda')
        model.load_state_dict(torch.load(f"./checkpoints/pavia/best_mymodel_x{self.scale}.pth"))
        return model.to(self.device)
    
    def init_3dfcn(self):
        model = d3fcn()
        model.load_state_dict(torch.load(f"./checkpoints/pavia/best_d3fcn_x{self.scale}.pth"))
        return model.to(self.device)
    
    def init_sspsr(self):
        n_subs = 8
        n_ovls = 1
        colors = 102
        n_blocks = 4
        n_feats = 256
        n_scale = self.scale

        model= SSPSR(n_subs=n_subs, n_ovls=n_ovls, n_colors=colors, n_blocks=n_blocks, n_feats=n_feats,
                            n_scale=n_scale, res_scale=0.1, use_share=True, conv=default_conv)
        model.load_state_dict(torch.load(f"./checkpoints/pavia/best_sspsr_x{self.scale}.pth"))
        return model.to(self.device)
    
    def init_gdrrn(self):
        model= GDRRN(input_chnl_hsi=102)
        model.load_state_dict(torch.load(f"./checkpoints/pavia/best_gdrrn_x{self.scale}.pth"))
        return model.to(self.device)
    def init_essaformer(self):
        model= ESSA(102, dim=256, upscale=self.scale)
        model.load_state_dict(torch.load(f"./checkpoints/pavia/best_essaformer_x{self.scale}.pth"))
        return model.to(self.device)
    def load_models(self):
        """预加载所有模型"""
        for name, init_fn in self.models.items():
            try:
                self.model_instances[name] = init_fn()
                self.model_instances[name].eval()
                print(f"{name} 模型加载成功")
            except Exception as e:
                print(f"{name} 加载失败: {str(e)}")
    def generate_random_hsi(self):
        """生成随机的高光谱数据 (1, 102, 224, 224) 并随机选择3个通道"""
        file_name = './data/Pavia.mat'
        data = sio.loadmat(file_name)['pavia']
        data = np.transpose(data, [2, 0, 1]).astype(np.float32)
        h=random.randint(0,data.shape[1]-self.patch_size)
        w=random.randint(0,data.shape[2]-self.patch_size)
        hsi_data = data[:, h:h + self.patch_size, w:w + self.patch_size]
        self.selected_channels = sorted(random.sample(range(102), 3))
        self.hsi_data=hsi_data
        return hsi_data  # 添加batch维度
    
    def get_rgb_image(self, hsi_data):
        """从高光谱数据中提取选定的3个通道作为RGB图像"""
        if self.selected_channels is None:
            self.selected_channels = sorted(random.sample(range(102), 3))
        
        # 归一化并转换为8-bit图像
        rgb_data = hsi_data[0, self.selected_channels, :, :]  # (3, H, W)
        rgb_data = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min()) * 255
        rgb_data = rgb_data.transpose(1, 2, 0).astype(np.uint8)  # (H, W, 3)
        return rgb_data
    def show(self):
        # 生成随机高光谱数据 (102, 224, 224)
            hr = self.generate_random_hsi()
            # 获取显示的RGB图像
            hr_rgb = self.get_rgb_image(hr[np.newaxis, :])
            upscale_factor=self.scale
            filtered_data = np.zeros_like(hr, dtype=np.float32)
            for i in range(hr.shape[0]):
                filtered_data[i, :, :] = gaussian_filter(hr[i, :, :], sigma=math.sqrt(0.72), radius=2)

            # 平均池化下采样
            downsampled_data = np.zeros(
                (hr.shape[0], hr.shape[1] // upscale_factor*2, hr.shape[2] // upscale_factor*2))
            for i in range(hr.shape[0]):
                downsampled_data[i, :, :] = cv2.resize(filtered_data[i, :, :],
                                                    (downsampled_data.shape[2], downsampled_data.shape[1]),
                                                    interpolation=cv2.INTER_CUBIC)
            self.model_hr=torch.from_numpy(downsampled_data).float().to(self.device).unsqueeze(0)
            lr = np.zeros(
                (hr.shape[0], hr.shape[1] // upscale_factor, hr.shape[2] // upscale_factor))
            for i in range(hr.shape[0]):
                lr[i, :, :] = cv2.resize(filtered_data[i, :, :],
                                                    (lr.shape[2], lr.shape[1]),
                                                    interpolation=cv2.INTER_CUBIC)
            bicubic_hr = np.zeros_like(filtered_data)
            for i in range(filtered_data.shape[0]):
                bicubic_hr[i, :, :] = cv2.resize(lr[i, :, :],
                                                    (filtered_data.shape[2], filtered_data.shape[1]),
                                                    interpolation=cv2.INTER_CUBIC)
            lr=torch.from_numpy(lr).float().to(self.device).unsqueeze(0)
            bicubic_hr=torch.from_numpy(bicubic_hr).float().to(self.device).unsqueeze(0)
            hr=torch.from_numpy(hr).float().to(self.device).unsqueeze(0)
            self.lr=lr
            self.bicubic_hr=bicubic_hr
            bicubic_hr_np=bicubic_hr.detach().cpu().numpy()[0]
            lr_np=lr.detach().cpu().numpy()[0]
            bicubichr_rgb=self.get_rgb_image(bicubic_hr_np[np.newaxis, :])
            lr_rgb=self.get_rgb_image(lr_np[np.newaxis, :])
            # 保存输入和输出图像
            hr_path = os.path.join( f"./gradio_image/paviahr224*224.png")
            lr_path=os.path.join( f"./gradio_image/pavialr224*224x{self.scale}.png")
            bicubic_path=os.path.join( f"./gradio_image/pavia_bicubichrx{self.scale}.png")
            cv2.imwrite(hr_path, cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(bicubic_path, cv2.cvtColor(bicubichr_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(lr_path, cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2BGR))
            return hr_rgb,lr_rgb,bicubichr_rgb
    def predict(self, model_name):
        """统一预测接口"""
        if model_name not in self.model_instances:
            return None, f"模型 {model_name} 未加载"
       
            
        
        lr=self.lr
        bicubic_hr=self.bicubic_hr
            # 各模型专用处理
        if model_name == "4MSCRN" or model_name == "essaformer":
            output = self.model_instances[model_name](lr)
            #output = self.model_hr
        elif model_name == "3dfcn":
            output = self.model_instances[model_name](bicubic_hr.unsqueeze(1)).squeeze(1)
        elif model_name == "sspsr":
            output = self.model_instances[model_name](lr,bicubic_hr)
        elif model_name == "gdrrn":
            output = self.model_instances[model_name](bicubic_hr)
            # 从输出中提取相同的3个通道
        output_np = output.detach().cpu().numpy()[0]  # (102, H, W)
          
        output_rgb = self.get_rgb_image(output_np[np.newaxis, :])  # 添加batch维度
           
        output_path = os.path.join( f"./gradio_image/pavia_sr_{model_name}x{self.scale}.png")
        cv2.imwrite(output_path, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
        return output_rgb
  
    

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import random
import math
from scipy.ndimage import gaussian_filter
from tkinter import font

class HSIViewerApp:
    def __init__(self, root,scale=4,patch_size=224):
        self.root = root
        self.root.title("Hyperspectral image super-resolution system")
        self.root.geometry("1400x1000")
        self.scale=scale
        self.patch_size=patch_size
        # 初始化模型
        self.hsi_models = HSI_Models(upscale=scale,patch_size=patch_size)
        self.hsi_models.load_models()
        # 创建界面组件
        self.create_widgets()
        
        # 生成初始数据
        self.generate_new_data()
    
    def create_widgets(self):
        """创建所有界面组件"""
        # 顶部控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=20)
        
        # 刷新数据按钮
        self.refresh_btn = tk.Button(
            control_frame, 
            text="Generate new data", 
            command=self.generate_new_data,
            font=('Arial', 12),
            height=2,
            width=15,
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=20)
        
        # 模型选择下拉菜单
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=list(self.hsi_models.models.keys()),
            font=('Arial', 12),
            width=15,
            state="readonly"
        )
        self.model_combobox.pack(side=tk.LEFT, padx=20)
        self.model_combobox.set("3dfcn")
        
        # 超分辨率按钮
        self.process_btn = tk.Button(
            control_frame, 
            text="Perform super-resolution", 
            command=self.process_image,
            font=('Arial', 12),
            height=2,
            width=15
        )
        self.process_btn.pack(side=tk.LEFT, padx=20)
        
        # 图像展示区域
        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 第一行：原始图像和下采样图像
        row1_frame = tk.Frame(image_frame)
        row1_frame.pack(fill=tk.X, pady=10)
        
        # 原始图像
        original_frame = tk.Frame(row1_frame)
        original_frame.pack(side=tk.LEFT, expand=True)
        tk.Label(original_frame, text=f"Original image({self.patch_size}×{self.patch_size})", font=('Arial', 12)).pack()
        self.original_img_label = tk.Label(original_frame)
        self.original_img_label.pack()
        
        # 下采样图像
        downsampled_frame = tk.Frame(row1_frame)
        downsampled_frame.pack(side=tk.LEFT, expand=True)
        tk.Label(downsampled_frame, text=f"Downsample the image ({self.patch_size/self.scale}×{self.patch_size/self.scale})", font=('Arial', 12)).pack()
        self.downsampled_img_label = tk.Label(downsampled_frame)
        self.downsampled_img_label.pack()
        
        # 第二行：Bicubic恢复和超分辨率结果
        row2_frame = tk.Frame(image_frame)
        row2_frame.pack(fill=tk.X, pady=10)
        
        # Bicubic恢复图像
        bicubic_frame = tk.Frame(row2_frame)
        bicubic_frame.pack(side=tk.LEFT, expand=True)
        tk.Label(bicubic_frame, text=f"Bicubic_up ({self.patch_size}×{self.patch_size})", font=('Arial', 12)).pack()
        self.bicubic_img_label = tk.Label(bicubic_frame)
        self.bicubic_img_label.pack()
        
        # 超分辨率结果
        result_frame = tk.Frame(row2_frame)
        result_frame.pack(side=tk.LEFT, expand=True)
        tk.Label(result_frame, text=f"SR_result ({self.patch_size}×{self.patch_size})", font=('Arial', 12)).pack()
        self.result_img_label = tk.Label(result_frame)
        self.result_img_label.pack()
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("ready")
        self.status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var,
            bd=1, 
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Arial', 20)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def generate_new_data(self):
        """生成新的高光谱数据并显示"""
        self.status_var.set("New hyperspectral data is being generated...")
        self.root.update()
        
        try:
            # 生成随机高光谱数据并获取所有图像
            hr_rgb, lr_rgb, bicubic_rgb= self.hsi_models.show()
            
            # 显示图像
            self.show_image(hr_rgb, self.original_img_label, f" Original image({self.patch_size}×{self.patch_size})")
            self.show_image(lr_rgb, self.downsampled_img_label, f"Downsample the image({self.patch_size/self.scale}×{self.patch_size/self.scale})")
            self.show_image(bicubic_rgb, self.bicubic_img_label, f"Bicubic_up ({self.patch_size}×{self.patch_size})")
            
            # 清空结果图像
            self.result_img_label.config(image=None)
            self.result_img_label.image = None
            
            self.status_var.set("Once the data is generated, select the model and perform the super-resolution")
        except Exception as e:
            self.status_var.set(f"Data generation failed: {str(e)}")
    
    def process_image(self):
        """执行超分辨率处理"""
        model_name = self.model_var.get()
        if not model_name:
            self.status_var.set("Please select the model first")
            return
        
        self.status_var.set(f"Super-resolution processing with {model_name}...")
        self.root.update()
        
        try:
            # 调用模型处理
            output_rgb = self.hsi_models.predict(model_name)
            
            if output_rgb is not None:
                self.show_image(output_rgb, self.result_img_label, f"Super-resolution results ({self.patch_size}×{self.patch_size})")
                self.status_var.set(f"{model_name} Super-resolution processing is complete")
            else:
                self.status_var.set("超分辨率处理失败: 无输出结果")
        except Exception as e:
            self.status_var.set(f"处理失败: {str(e)}")
    
    def show_image(self, cv2_img, label_widget, title=None):
        """安全显示图像函数（带错误处理）"""
        try:
            # 验证输入
            if cv2_img is None:
                raise ValueError("图像数据为空")
                
            if not isinstance(cv2_img, np.ndarray):
                print(cv2_img)
                raise ValueError(f"需要numpy数组，得到{type(cv2_img)}")
                
            if cv2_img.dtype != np.uint8:
                cv2_img = cv2_img.astype(np.uint8)
                
            # 确保是3通道彩色图像
            if len(cv2_img.shape) == 2:  # 灰度图
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
            elif cv2_img.shape[2] == 4:  # 带alpha通道
                cv2_img = cv2_img[:, :, :3]
                
            # 转换颜色空间
            img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL Image
            pil_img = Image.fromarray(img_rgb)
            
            # 调整大小
            display_width = 400
            w_percent = display_width / float(pil_img.size[0])
            h_size = int(float(pil_img.size[1]) * float(w_percent))
            pil_img = pil_img.resize((display_width, h_size), Image.LANCZOS)
            
            # 显示图像
            tk_img = ImageTk.PhotoImage(pil_img)
            label_widget.config(image=tk_img)
            label_widget.image = tk_img
            
            if title:
                label_widget.master.children['!label'].config(text=title)
                
        except Exception as e:
            print(f"显示图像错误: {str(e)}")
            # 显示错误占位图
            error_img = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(error_img, f"Error: {str(e)}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            self.show_image(error_img, label_widget, "图像加载失败")
                
        

# 主程序
if __name__ == "__main__":
    
    root = tk.Tk()
    app = HSIViewerApp(root,8,256)
    root.mainloop()
