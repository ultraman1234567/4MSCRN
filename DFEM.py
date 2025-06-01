import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from SFEM import ChannelAttention,SpatialAttention,CBAM
from Patch_split import *
from SS_MambaBlock import SS1D_MB,SS2D_MB,SS3D_MB
from scpa import SCPA
class Mlp(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.LeakyReLU(), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight,  mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpaFEM(nn.Module):
    
    def __init__(self, input_shape,channels,#c
                 patch_size,#每个patch大小
                 embed_dim, #每个patch的向量维度
                 CA_ratio=16,
                 norm_layer=nn.BatchNorm2d ,
                 mlp_ratio=4., drop=0., act_layer=nn.LeakyReLU(), 
                 SS2D_MBs_num=8,#ss2d_mb的数目
                #---------------------SS2D_MB参数---------------------
            
            
            
            
                    d_state=16,#SSM维度
                    #d_state="auto", # 20240109
                    d_conv=3,
                    expand=2,
                    dt_rank="auto",
                    dt_min=0.001,
                    dt_max=0.1,
                    dt_init="random",
                    dt_scale=1.0,
                    dt_init_floor=1e-4,
                    dropout=0.,
                    laynorm=nn.LayerNorm,
                    conv_bias=True,
                    bias=False,
                    merge="learnable",
                    device=None,
                    dtype=None,
                #-----------------------------------------------------
                 ):
        super().__init__()
        self.norm1 = norm_layer(channels)
        #patch
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.device=device
        
        self.CA=ChannelAttention(channels,CA_ratio)
        #self.SS2D_MB=SS2D_MB(embed_dim,d_state,d_conv,expand,dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,dropout,laynorm,conv_bias,bias,merge,device,dtype)
        self.SS2D_MBs=nn.Sequential(*[SS2D_MB(embed_dim,d_state,d_conv,expand,dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,dropout,laynorm,conv_bias,bias,merge,device,dtype)
                                     for _ in range(SS2D_MBs_num)])
        self.norm2 = norm_layer(channels)
        """ mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        ) """
        self.scpa=SCPA(channels)
        self.patch_embed_2d=Patch_embedding_2d(self.patch_size,embed_dim,input_shape,'normal')
        self.patch_unembed_2d=Patch_embedding_2d(self.patch_size,embed_dim,input_shape,'inverse')
    def forward(self, x):
        # 输入：(b,c,h,w) 输出：(b,c,h,w)
        residual=x
        #x=x.permute(0,2,3,1).contiguous()#(b,h,w,c)
        #x=self.norm1(x).permute(0,3,1,2).contiguous()#(b,c,h,w)
        x=self.norm1(x)
        x1 = self.CA(x)#(b,c,h,w)
        #print(x.shape)
        x2 = self.patch_embed_2d(x)
        #print(x2.shape)
        #for layer in self.SS2D_MBs:
        #    x2=layer(x2)
        x2=self.SS2D_MBs(x2)
        #print(x2.shape)
        x2=self.patch_unembed_2d(x2)#(b,c,h,w)
        x=residual+x1+x2#(b,c,h,w)
        residual2=x
        #x=x.permute(0,2,3,1).contiguous()#(b,h,w,c)
        # FFN子层
        #x =  self.mlp(self.norm2(x)).permute(0,3,1,2).contiguous()#(b,c,h,w)
        x=self.scpa(self.norm2(x))
        return x+residual2

""" if __name__ == "__main__":
    x=torch.randn(1,288,100,100).to('cuda')
    spafem=SpaFEM(288,(20,20),300,device='cuda').to('cuda')
    print(spafem(x).shape) """

class SpeFEM(nn.Module):
    
    def __init__(self, input_shape,channels,#c
                 embed_dim, #每个光谱的向量维度
                 SA_kernel_size=7,
                 norm_layer=nn.BatchNorm2d, 
                 mlp_ratio=4., drop=0., act_layer=nn.LeakyReLU(), 
                 
                 SS1D_MBs_num=8,#SS1D_MB数目
                #---------------------SS2D_MB参数---------------------
            
                    d_state=16,#SSM维度
                    #d_state="auto", # 20240109
                    d_conv=3,
                    expand=2,
                    dt_rank="auto",
                    dt_min=0.001,
                    dt_max=0.1,
                    dt_init="random",
                    dt_scale=1.0,
                    dt_init_floor=1e-4,
                    dropout=0.,
                    laynorm=nn.LayerNorm,
                    conv_bias=True,
                    bias=False,
                    if_bid=True,
                    merge="learnable",
                    device=None,
                    dtype=None,
                #-----------------------------------------------------
                 ):
        super().__init__()
        self.norm1 = norm_layer(channels)
        #patch_split
        self.embed_dim=embed_dim
        self.device=device
        #
        self.SA=SpatialAttention(SA_kernel_size)
        #self.SS1D_MB=SS1D_MB(embed_dim,d_state,d_conv,expand,dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,dropout,laynorm,conv_bias,bias,if_bid,merge,device,dtype)
        self.SS1D_MBs=nn.Sequential(*[SS1D_MB(embed_dim,d_state,d_conv,expand,dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,dropout,laynorm,conv_bias,bias,if_bid,merge,device,dtype)
                                     for _ in range(SS1D_MBs_num)])
        self.norm2 = norm_layer(channels)
        """ mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        ) """
        self.scpa=SCPA(channels)
        self.patch_embed_1d=Patch_embedding_1d(embed_dim,input_shape,'normal')
        self.patch_unembed_1d=Patch_embedding_1d(embed_dim,input_shape,'inverse')
    def forward(self, x):
        # 输入：(b,c,h,w) 输出：(b,c,h,w)
        residual=x
        #x=x.permute(0,2,3,1).contiguous()#(b,h,w,c)
        #x=self.norm1(x).permute(0,3,1,2).contiguous()#(b,c,h,w)
        x=self.norm1(x)
        x1 = self.SA(x)#(b,c,h,w)
        x2=self.patch_embed_1d(x)
        #print(x2.shape)
        #for layer in self.SS1D_MBs:
        x2 = self.SS1D_MBs(x2)
        #print(x2.shape)
        x2=self.patch_unembed_1d(x2)#(b,c,h,w)
        x=residual+x1+x2#(b,c,h,w)
        residual2=x
        #x=x.permute(0,2,3,1).contiguous()#(b,h,w,c)
        # FFN子层
        #x =  self.mlp(self.norm2(x)).permute(0,3,1,2).contiguous()#(b,c,h,w)
        x=self.scpa(self.norm2(x))
        return x+residual2
""" if __name__ == "__main__":
    x=torch.randn(1,288,100,100).to('cuda')
    spefem=SpeFEM(288,500,7,device='cuda').to('cuda')
    print(spefem(x).shape) """
    
class _3DLPFEM(nn.Module):
    
    def __init__(self, input_shape,channels,#c
                 patch_size,#每个patch大小
                 embed_dim, #每个patch的向量维度
                 CA_ratio=16,
                 SA_kernel_size=7,
                 norm_layer=nn.BatchNorm2d, 
                 mlp_ratio=4., drop=0., act_layer=nn.LeakyReLU(), 
                 
                 SS3D_MBs_num=8,
                #---------------------SS2D_MB参数---------------------
            
                    d_state=16,#SSM维度
                    #d_state="auto", # 20240109
                    d_conv=3,
                    expand=2,
                    dt_rank="auto",
                    dt_min=0.001,
                    dt_max=0.1,
                    dt_init="random",
                    dt_scale=1.0,
                    dt_init_floor=1e-4,
                    dropout=0.,
                    laynorm=nn.LayerNorm,
                    conv_bias=True,
                    bias=False,
                    merge="learnable",
                    device=None,
                    dtype=None,
                #-----------------------------------------------------
                 ):
        super().__init__()
        self.norm1 = norm_layer(channels)
        #patch
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.device=device
        
        self.CBAM=CBAM(channels,CA_ratio,SA_kernel_size)
        #self.SS3D_MB=SS3D_MB(embed_dim,d_state,d_conv,expand,dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,dropout,laynorm,conv_bias,bias,merge,device,dtype)
        self.SS3D_MBs= nn.Sequential(*[SS3D_MB(embed_dim,d_state,d_conv,expand,dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,dropout,laynorm,conv_bias,bias,merge,device,dtype)
                                      for _ in range(SS3D_MBs_num)])
        self.norm2 = norm_layer(channels)
        """ mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        ) """
        self.scpa=SCPA(channels)
        self.patch_embed_3d=Patch_embedding_3d(self.patch_size,embed_dim,input_shape,'normal')
        self.patch_unembed_3d=Patch_embedding_3d(self.patch_size,embed_dim,input_shape,'inverse')
    def forward(self, x):
        # 输入：(b,c,h,w) 输出：(b,c,h,w)
        residual=x
        #x=x.permute(0,2,3,1).contiguous()#(b,h,w,c)
        #x=self.norm1(x).permute(0,3,1,2).contiguous()#(b,c,h,w)
        #x=self.norm1(x)
        x1 = self.CBAM(x)#(b,c,h,w)
        x2 = self.patch_embed_3d(x)
        #for layer in self.SS3D_MBs:
        x2=self.SS3D_MBs(x2)
        x2=self.patch_unembed_3d(x2)#(b,c,h,w)
        x=residual+x1+x2#(b,c,h,w)
        residual2=x
        #x=x.permute(0,2,3,1).contiguous()#(b,h,w,c)
        # FFN子层
        #x =  self.mlp(self.norm2(x)).permute(0,3,1,2).contiguous()#(b,c,h,w)
        x=self.scpa(self.norm2(x))
        return x+residual2
    
""" if __name__ == "__main__":
    x=torch.randn(1,288,100,100).to('cuda')
    _3dlpfem=_3DLPFEM(288,(4,20,20),1000,device='cuda').to('cuda')
    print(_3dlpfem(x).shape) """
    



import copy
class DFEM_block(nn.Module):
    def __init__(self, SpaFEM_N1,SpeFEM_N2,SpaFEM,SpeFEM,conv_inchannels,_3DLPFEM,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SpaFEM=nn.Sequential(*[copy.deepcopy(SpaFEM) for _ in range(SpaFEM_N1)])
        self.SpeFEM=nn.Sequential(*[copy.deepcopy(SpeFEM) for _ in range(SpeFEM_N2)])
        self.conv2d=nn.Sequential(nn.Conv2d(2*conv_inchannels,conv_inchannels,kernel_size=3,padding=1),
                                  nn.BatchNorm2d(conv_inchannels))
        self._3DLPFEM=_3DLPFEM
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self,x):
        residual=x
        x1=x
        x2=x
        for layer1 in self.SpaFEM:
            x1=layer1(x1)
        x1=residual+x1
        for layer2 in self.SpeFEM:
            x2=layer2(x2)
        x2=residual+x2
        x=torch.cat([x1,x2],dim=1)
        x=self.conv2d(x)
        x=self._3DLPFEM(x)
        return residual+x

class DFEM(nn.Module):
    def __init__(self, 
                 SpaFEM_N1,SpeFEM_N2,SpaFEM,SpeFEM,conv_inchannels,_3DLPFEM,#DFEM_block
                 DFEM_block_N3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.DFEM_block=DFEM_block(SpaFEM_N1,SpeFEM_N2,SpaFEM,SpeFEM,conv_inchannels,_3DLPFEM)
        self.DFEM_blocks=nn.Sequential(*[DFEM_block(SpaFEM_N1,SpeFEM_N2,SpaFEM,SpeFEM,conv_inchannels,_3DLPFEM) for _ in range(DFEM_block_N3)])
    
    def forward(self,x):
        residual=x
        #for layer in self.DFEM_blocks:
        x=self.DFEM_blocks(x)
        return residual+x
        
if __name__ == "__main__":
    x=torch.randn(16,16,64,64).to('cuda')
    spafem=SpaFEM(x.shape,16,(2,2),15,device='cuda').to('cuda')
    spefem=SpeFEM(x.shape,16,256,device='cuda').to('cuda')
    _3dlpfem=_3DLPFEM(x.shape,16,(2,8,8),100,device='cuda').to('cuda')
    dfem=DFEM(2,1,spafem,spefem,16,_3dlpfem,1).to('cuda')
    print(dfem(x).shape)