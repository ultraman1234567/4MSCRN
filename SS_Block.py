import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    
class SS1D_block(nn.Module):
    '''
    输入x:(B,L,D),输入y:(B,L,D)
    '''
    def __init__(self,
        d_state=16,#SSM状态维度
        d_inner=98,#内部拓展的向量维度对应D
        dt_rank=4,#mamba自带的dt初始化
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        #---------------------
        if_bid=True,
        #----------------------
        merge="learnable",#合并方式，目前：learnable，sum
        #---------------------------
        device=None,
        dtype=None,
        **kwargs,):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank=dt_rank
        #初始化 B，C，dt，用到的x与相关
        if(if_bid):
            self.K=2
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            )
            self.x_proj_weight_bid = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=2, N, inner)
            del self.x_proj

            self.dt_projs = (
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            )
            self.dt_projs_weight_bid = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=2, inner, rank)
            self.dt_projs_bias_bid = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=2, inner)
            del self.dt_projs
        else:
            self.K=1
            self.x_proj_weight_one=nn.Parameter(nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight.unsqueeze(0))
            self.dt_projs_weight_one=nn.Parameter(self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs).weight.unsqueeze(0))
            self.dt_projs_bias_one = nn.Parameter(self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs).bias.unsqueeze(0))

        #初始化A,D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True) # (K, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True) # (k, D, N)
        
        
        #扫描函数
        self.if_bid=if_bid
        self.selective_scan = selective_scan_fn
        self.forward_core = self.Selective_scan
        #合并四方向
        self.merge=merge
        self.mergeconv = nn.Conv2d(self.K, 1, kernel_size=1)
    
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def Selective_scan(self, x: torch.Tensor):
        # 输入X：(B,L,D) ，输出:(B,L,D)
        B, L, D = x.shape
        x_proj_weight=self.x_proj_weight_bid
        dt_projs_weight=self.dt_projs_weight_bid
        dt_projs_bias = self.dt_projs_bias_bid
        if(self.if_bid):
            K = 2
            x=x.permute(0,2,1).contiguous()#(b,d,l)
            x_reversed = torch.flip(x, dims=[-1]) 
            xs = torch.stack([x, x_reversed], dim=1)
             # (b, k, d, l)
        else:
            K=1
            xs=x.permute(0,2,1).contiguous().unsqueeze(1) # (b, k, d, l)
            x_proj_weight=self.x_proj_weight_one
            dt_projs_weight=self.dt_projs_weight_one
            dt_projs_bias = self.dt_projs_bias_one
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        
        if(self.if_bid):
            y_original = out_y[:, 0, :, :]  # 形状 (b, d, l)
            y_reversed = out_y[:, 1, :, :]  # 形状 (b, d, l)
            y_reversed_recovered = torch.flip(y_reversed, dims=[-1])
            return y_original.permute(0,2,1).contiguous(),y_reversed_recovered.permute(0,2,1).contiguous()
        else:
            return out_y.squeeze(1).permute(0,2,1).contiguous()
    def forward(self, x: torch.Tensor, **kwargs):
        #输入X：(b,l,d) ，y:(b,l,d) 
        if(self.if_bid):
            y1,y2=self.forward_core(x)
            assert y1.dtype == torch.float32
            if(self.merge=="sum"):
                y = y1 + y2 
            elif(self.merge=="learnable"):    
                #方案二：可学习参数融合使用1*1卷积：
                y=torch.stack([y1, y2], dim=1)# (B,2,L,D)
                y=self.mergeconv(y).squeeze(1)
        else:
            y=self.forward_core(x)
        return y

""" if __name__ == '__main__':
    ss2d = SS1D_block(d_inner=15,if_bid=True).cuda()
    x = torch.randn(1, 9999, 15) 
    x = x.cuda()
    y = ss2d(x)
    print(y.shape) """


class SS2D_block(nn.Module):
    """输入X：(B,H,W,D) H,W表示水平垂直分别有多少patch，D表示每个patch的向量，输出Y(B,H,W,D)

        d_state=16,#SSM状态维度
        d_inner=98,#内部拓展的向量维度,对应D
        dt_rank=4,#mamba自带的dt初始化
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        #----------------------
        merge="learnable",#合并方式，目前：learnable，sum
        
        四个方向的扫描
    """
    def __init__(
        self,
        d_state=16,#SSM状态维度
        d_inner=98,#内部拓展的向量维度对应D
        dt_rank=4,#mamba自带的dt初始化
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        #---------------------
        #----------------------
        merge="learnable",#合并方式，目前：learnable，sum
        #---------------------------
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank=dt_rank
        #初始化 B，C，dt，用到的x与相关
        #四方向
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        self.K=4
        
        
        
        #初始化A,D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True) # (K, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True) # (k, D, N)
        
        
        #扫描函数
        self.selective_scan = selective_scan_fn
        self.forward_core = self.Selective_scan
        #合并四方向
        self.merge=merge
        self.mergeconv = nn.Conv2d(self.K, 1, kernel_size=1)

        
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def Selective_scan(self, x: torch.Tensor):
        # 输入X：(B,H,W,D) H,W表示水平垂直分别有多少patch，D表示每个patch的向量，
        B, H, W, D = x.shape
        L = H * W
        x=x.permute(0,3,1,2).contiguous()
        x_proj_weight=self.x_proj_weight
        dt_projs_weight=self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias

        K = 4
            #四个方向的序列xs
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y# 形状 (b, d, l)
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self, x: torch.Tensor, **kwargs):
        #输入X：(B,H,W,D) H,W表示水平垂直分别有多少patch，C表示每个patch的向量，y:(b,h,w,d)
        B,  H, W, D = x.shape
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
            #方案一：sum
        if(self.merge=="sum"):
            y = y1 + y2 + y3 + y4
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        elif(self.merge=="learnable"):    
                #方案二：可学习参数融合使用1*1卷积：
            y=torch.stack([y1, y2, y3, y4], dim=1)# (B,4,D,L)
            y=self.mergeconv(y)
            y=y.squeeze(1).view(B,-1, H, W).permute(0,2,3,1)
        return y
""" if __name__ == '__main__':
    ss2d = SS2D_block(d_inner=15).cuda()
    x = torch.randn(1, 80,64,15) 
    x = x.cuda()
    y = ss2d(x)
    print(y.shape) """

class SS3D_block(nn.Module):
    """输入X：(B,H,W,Len,D) H,W,Len表示长宽高分别有多少patch，D表示每个patch的向量，输出Y(B,H,W,Len,D)

        d_state=16,#SSM状态维度
        d_inner=98,#内部拓展的向量维度,对应D
        dt_rank=4,#mamba自带的dt初始化
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        #----------------------
        merge="learnable",#合并方式，目前：learnable，sum
        
        六个方向的扫描
    """
    def __init__(
        self,
        d_state=16,#SSM状态维度
        d_inner=98,#内部拓展的向量维度对应D
        dt_rank=4,#mamba自带的dt初始化
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        #----------------------
        merge="learnable",#合并方式，目前：learnable，sum
        #---------------------------
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank=dt_rank
        #初始化 B，C，dt，用到的x与相关
        #六方向
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=12, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=12, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=12, inner)
        del self.dt_projs
        self.K=12
        
        
        
        #初始化A,D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True) # (K, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True) # (k, D, N)
        
        
        #扫描函数
        self.selective_scan = selective_scan_fn
        self.forward_core = self.Selective_scan
        #合并四方向
        self.merge=merge
        self.mergeconv = nn.Conv2d(self.K, 1, kernel_size=1)
    @staticmethod
    def ranking_3Dpatch_seq(x):
        """
    Args:
        x: Input tensor of shape (B, L, W, H, D)
    
    Returns:
        Tensor of shape (B, 12, L*W*H, D) containing all 12 sequences
    """
        B, L, W, H, D = x.shape
        LEN = L * W * H
    
        # Initialize result tensor
        result = torch.zeros(B, 12, LEN, D, device=x.device)
        
        # Generate all 6 possible dimension orders
        dim_orders = [
            ['L', 'W', 'H'], ['L', 'H', 'W'],
            ['H', 'W', 'L'], ['H', 'L', 'W'],
            ['W', 'L', 'H'], ['W', 'H', 'L']
        ]
        
        for i, dim_order in enumerate(dim_orders):
            # Get the corresponding dimensions
            dims = []
            for d in dim_order:
                if d == 'L':
                    dims.append(1)  # L is at dim 1
                elif d == 'W':
                    dims.append(2)  # W is at dim 2
                elif d == 'H':
                    dims.append(3)  # H is at dim 3
            
            # Forward sequence
            permute_dims = [0] + dims + [4]
            reshaped = x.permute(*permute_dims).reshape(B, -1, D)
            result[:, 2*i, :, :] = reshaped
            
            # Backward sequence: use torch.flip ,反转对应维度
            reversed_x = torch.flip(x, dims=dims)  # 
            reshaped_rev = reversed_x.permute(*permute_dims).reshape(B, -1, D)
            result[:, 2*i+1, :, :] = reshaped_rev
        
        return result
    @staticmethod
    def restore_originalseq(output, original_shape):
        """
        Args:
            output: Tensor of shape (B, 12, Len, D) from the neural network.
            original_shape: Tuple (B, L, W, H, D) of the original tensor.
        
        Returns:
            List of 12 tensors, each of shape (B, L, W, H, D).
        """
        B, L, W, H, D = original_shape
        Len = L * W * H
        
        # Define the 6 dimension orders (same as in generate_sequences)
        dim_orders = [
            ['L', 'W', 'H'], ['L', 'H', 'W'],
            ['H', 'W', 'L'], ['H', 'L', 'W'],
            ['W', 'L', 'H'], ['W', 'H', 'L']
        ]
        restored_sequences = []
        
        for i in range(12):
            # Get the sequence (B, Len, D)
            seq = output[:, i, :, :]
            
            # Determine if it's forward (even i) or backward (odd i)
            is_backward = (i % 2 == 1)
            dim_order = dim_orders[i // 2]  # Get the corresponding dim_order
            # Map 'L', 'W', 'H' to their original dimensions (1, 2, 3)
            dims = []
            dims_shape=[]
            inv_permute = [0]  # batch dim
            for d in dim_order:
                if d == 'L':
                    dims.append(1)
                    inv_permute.append(1)
                    dims_shape.append(L)
                elif d == 'W':
                    dims.append(2)
                    inv_permute.append(2)
                    dims_shape.append(W)
                elif d == 'H':
                    dims.append(3)
                    inv_permute.append(3)
                    dims_shape.append(H)
            inv_permute.append(4)  # feature dim (D)
            
            temp_shape = [B] + dims_shape + [D]
            reshaped = seq.reshape(temp_shape)
            
        
            original_permute = [0] + dims + [4]
            inv_permute = [0] * 5
            for idx, pos in enumerate(original_permute):
                inv_permute[pos] = idx
            restored = reshaped.permute(*inv_permute)
            
            # If it was a backward scan, flip back the dimensions
            if is_backward:
                restored = torch.flip(restored, dims=dims)
            
            restored_sequences.append(restored)
        
        return restored_sequences
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def Selective_scan(self, x: torch.Tensor):
        # 输入X：(B,H,W,Len,D) H,W，Len表示长宽高分别有多少patch，D表示每个patch的向量，
        B, H, W,Len , D = x.shape
        L = H * W * Len
        
        x_proj_weight=self.x_proj_weight
        dt_projs_weight=self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias

        K = 12
        #12个方向的序列xs
        xs=self.ranking_3Dpatch_seq(x).permute(0,1,3,2).contiguous()
         # (b, k, d, l)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        out_y=self.restore_originalseq(out_y,(B, H, W,Len , D))

        return out_y# 形状(b,H,W,Len,D)的张量的list
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self, x: torch.Tensor, **kwargs):
        #输入X：(b,H,W,Len,D) H,W表示长宽高分别有多少patch，D表示每个patch的向量，y:(b,H,W,Len,D)
        B,H,W,Len,D = x.shape
        L=H*W*Len
        y = self.forward_core(x)
        assert y[0].dtype == torch.float32
            #方案一：sum
        if(self.merge=="sum"):
            merged = torch.zeros_like(y[0])  # 初始化全零张量
            for tensor in y:
                merged += tensor  # 逐元素相加
            y = merged
        elif(self.merge=="learnable"):    
            #方案二：可学习参数融合使用1*1卷积：
            y=torch.stack(y, dim=1)# (B,12,H,W,Len,D)
            y=y.view(B,12,-1,D)
            y=self.mergeconv(y)
            y=y.squeeze(1).view(B,H,W,Len,D)
        return y
if __name__ == '__main__':
    ss3d = SS3D_block(d_inner=15).cuda()
    x = torch.randn(1, 12,80,64,15) 
    x = x.cuda()
    y = ss3d(x)
    print(y.shape)