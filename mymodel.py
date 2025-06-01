import torch
import torch.nn as nn
import torch.nn.functional as F
from SFEM import SFEM
from DFEM import DFEM,SpaFEM,SpeFEM,_3DLPFEM
from Up import ProgressiveSubPixelUpsampler,Net
class xxx(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self,x):
        return x
class xx(nn.Module):
    def __init__(self, channels,hidden,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.co=nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
    def forward(self,x):
        return self.co(x)
class mymodel(nn.Module):
    def __init__(self, 
                 hidd_channels,#中间光谱
                 channels,#光谱
                 SFEM,
                 DFEM,
                 upscale_factor,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SFEM=SFEM
        self.DFEM=DFEM
        self.upscale=upscale_factor
        self.Upsampe=Net(hidd_channels,hidd_channels,channels,upscale_factor,3,0)
        
    def forward(self,x):
        B,C,H,W=x.shape
        #print(x.shape)
        residual= F.interpolate(x,scale_factor=self.upscale,mode='bicubic',align_corners=True)
        x=self.SFEM(x)
        #print(x.shape)
        x=self.DFEM(x)
        x=self.Upsampe(x)
        return x+residual
class bicubic_up(nn.Module):
    def __init__(self,upscale ,*args, **kwargs):
        self.upscale=upscale
        super().__init__(*args, **kwargs)
    def forward(self,x):
        return  F.interpolate(x,scale_factor=self.upscale,mode='bicubic',align_corners=True)

