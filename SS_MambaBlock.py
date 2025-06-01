import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from SS_Block import SS1D_block,SS2D_block,SS3D_block

class SS2D_MB(nn.Module):
    def __init__(
        self,
        d_model,#每个patch的向量维度
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
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        #self.d_state = d_state
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_norm=laynorm(d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.ss2D_block=SS2D_block(self.d_state,self.d_inner,self.dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,merge,device,dtype)
        
        self.out_norm = laynorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self, x: torch.Tensor, **kwargs):
        #输入：（b,h,w,d）d是每个patch的向量维度，多少个 输出：（b,h,w,d）
        residual=x
        B, H, W, D = x.shape

        xz = self.in_proj(self.in_norm(x))
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()# (b, d, h, w)
        x = self.act(self.conv2d(x)).permute(0,2,3,1)# (b, h, w, d)
        y=self.ss2D_block(x)
        try:
            y = self.out_norm(y)
        except:
            y = self.out_norm.to(torch.float32)(y).half()

        y = y * F.silu(z)
        try:
            out = self.out_proj(y)
        except:
            out = self.out_proj.to(torch.float32)(y).half()
        if self.dropout is not None:
            out = self.dropout(out)
        out = out+residual
        return out
    
""" if __name__ == '__main__':
    ss2d = SS2D_MB(d_model=12,merge="sum").cuda()
    x = torch.randn(1,  64, 64,12) 
    x = x.cuda()
    y = ss2d(x)
    print(y.shape) """

class SS3D_MB(nn.Module):
    def __init__(
        self,
        d_model,#每个patch的向量维度
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
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        #self.d_state = d_state
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_norm=laynorm(d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.ss3D_block=SS3D_block(self.d_state,self.d_inner,self.dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,merge,device,dtype)
        
        self.out_norm = laynorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self, x: torch.Tensor, **kwargs):
        #输入：（b,h,w,l,d）d是每个patch的向量维度，多少个 输出：（b,h,w,l,d）
        residual=x
        B, H, W,L, D = x.shape

        xz = self.in_proj(self.in_norm(x))
        x, z = xz.chunk(2, dim=-1) # (b, h, w,l, d)

        x = x.permute(0, 4, 1, 2,3).contiguous().view(B,self.d_inner,-1)# (b, d, h*w*l)
        x = self.act(self.conv1d(x)).permute(0,2,1).view(B,H,W,L,self.d_inner)# (b, h, w,l, d)
        y=self.ss3D_block(x)
        try:
            y = self.out_norm(y)
        except:
            y = self.out_norm.to(torch.float32)(y).half()

        y = y * F.silu(z)
        try:
            out = self.out_proj(y)
        except:
            out = self.out_proj.to(torch.float32)(y).half()
        if self.dropout is not None:
            out = self.dropout(out)
        out = out+residual
        return out

""" if __name__ == '__main__':
    ss3d = SS3D_MB(d_model=12,merge="sum").cuda()
    x = torch.randn(1,  64,64, 64,12) 
    x = x.cuda()
    y = ss3d(x)
    print(y.shape) """

class SS1D_MB(nn.Module):
    def __init__(
        self,
        d_model,#每个patch的向量维度
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
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        #self.d_state = d_state
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_norm=laynorm(d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.ss1D_block=SS1D_block(self.d_state,self.d_inner,self.dt_rank,dt_min,dt_max,dt_init,dt_scale,dt_init_floor,if_bid,merge,device,dtype)
        
        self.out_norm = laynorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self, x: torch.Tensor, **kwargs):
        #输入：（b,l,d）d是每个patch的向量维度，多少个 输出：（b,l,d）
        residual=x
        B, L, D = x.shape

        xz = self.in_proj(self.in_norm(x))
        x, z = xz.chunk(2, dim=-1) # (b,L, d)

        x = x.permute(0, 2, 1).contiguous()# (b, d, L)
        x = self.act(self.conv1d(x)).permute(0,2,1)# (b, L, d)
        y=self.ss1D_block(x)
        try:
            y = self.out_norm(y)
        except:
            y = self.out_norm.to(torch.float32)(y).half()

        y = y * F.silu(z)
        try:
            out = self.out_proj(y)
        except:
            out = self.out_proj.to(torch.float32)(y).half()
        if self.dropout is not None:
            out = self.dropout(out)
        out = out+residual
        return out
if __name__ == '__main__':
    ss1d = SS1D_MB(d_model=256).cuda()
    x = torch.randn(16,  64,256) 
    x = x.cuda()
    y = ss1d(x)
    print(y.shape)