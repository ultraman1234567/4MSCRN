import torch
import torch.nn as nn


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    nn.init.normal_(tensor, mean=mean, std=std)
    with torch.no_grad():
        tensor.clamp_(min=a, max=b)

class Patch_embedding_1d(nn.Module):
    def __init__(self,embed_dim,input_original_shape,act:str='normal',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act=act
        self.embed_dim=embed_dim
        self.input_original_shape=input_original_shape
        if act=='normal':
            self.proj = nn.Linear(self.input_original_shape[1]*self.input_original_shape[2], embed_dim)
            nn.init.kaiming_normal_(self.proj.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.zeros_(self.proj.bias)
        elif act=='inverse':
            self.proj = nn.Linear(embed_dim,self.input_original_shape[1]*self.input_original_shape[2])
            nn.init.kaiming_normal_(self.proj.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.zeros_(self.proj.bias)
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self,x):
        if self.act=='normal':
            #x:(b,c,h,w)
            B, C, H, W = x.shape
            L=C
            x=x.view(B,L,-1)
            x=self.proj(x)#(b,l,d)
            return x
        elif self.act=='inverse':
            #x:(b,c,emdim)
            C, H, W = self.input_original_shape
            L = C  # 1D patch 是按通道切分的

            # 线性投影回 H*W 空间
            x = self.proj(x)  # (B, L, H*W)

            # 恢复原始形状 (B, C, H, W)
            x = x.view(x.shape[0], C, H, W)
            return x
    
class Patch_embedding_2d(nn.Module):
    def __init__(self,patch_size, embed_dim,input_original_shape,act:str='normal',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size=patch_size
        self.act=act
        self.embed_dim=embed_dim
        self.input_original_shape=input_original_shape
        if act=='normal':
            P,Q = patch_size[0],patch_size[1]
            assert input_original_shape[1] % P == 0 and input_original_shape[2] % Q == 0, "H and W must be divisible by patch size"
            self.proj = nn.Conv2d(input_original_shape[0], embed_dim, kernel_size=(P,Q), stride=(P,Q))
            trunc_normal_(self.proj.weight, std=0.02)  # ViT 风格初始化
            nn.init.zeros_(self.proj.bias)  # bias 初始化为 0 
        elif act=='inverse':
            self.proj=nn.Linear(embed_dim, patch_size[0]*patch_size[1] * input_original_shape[0])
            nn.init.kaiming_normal_(self.proj.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.zeros_(self.proj.bias) 
    def forward(self,x):
        if self.act=='normal':
            x = self.proj(x)  # (B, D, h, w)
            x = x.permute(0, 2, 3, 1)  # (B, h, w, D)
            
            return x
        elif self.act=='inverse':
            #x:(b,h,w,emdim)
            B=x.shape[0]
            C, H, W = self.input_original_shape
            b,h,w,emdim=x.shape
            # 线性投影回 H*W 空间
            x = self.proj(x)  # (B, h,w, P*Q*C)
            x = x.reshape(B, h, w, self.patch_size[0],self.patch_size[1], C)
            x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, P, w, Q)
            x = x.reshape(B, C, H, W)  # 假设 h*P = H, w*Q = W
            return x              
            
class Patch_embedding_3d(nn.Module):
    def __init__(self,patch_size, embed_dim,input_original_shape,act:str='normal',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size=patch_size
        self.act=act
        self.embed_dim=embed_dim
        self.input_original_shape=input_original_shape
        if act=='normal':
            c, P, Q = patch_size[0], patch_size[1], patch_size[2]
            o =  self.input_original_shape[0]// c  # 通道维度 Patch 数
            assert self.input_original_shape[0] % c == 0 and self.input_original_shape[1] % P == 0 and self.input_original_shape[2] % Q == 0, "输入尺寸必须能被 Patch 大小整除"
            self.proj = nn.Conv2d(
        in_channels=self.input_original_shape[0],
        out_channels=o * embed_dim,
        kernel_size=(P, Q),
        stride=(P, Q),
        groups=o  # 分组卷积，每组处理 c 个通道
    )
            trunc_normal_(self.proj.weight, std=0.02)  # ViT 风格初始化
            nn.init.zeros_(self.proj.bias)  # bias 初始化为 0 
        elif act=='inverse':
            self.proj = nn.Linear(self.embed_dim, patch_size[0]*patch_size[1] *patch_size[2])
            nn.init.kaiming_normal_(self.proj.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.zeros_(self.proj.bias) 
    def forward(self,x):
        if self.act=='normal':
            c, P, Q = self.patch_size[0], self.patch_size[1], self.patch_size[2]
            B=x.shape[0]
            o =  self.input_original_shape[0]// c  # 通道维度 Patch 数
            h = self.input_original_shape[1] // P  # 空间高度 Patch 数
            w = self.input_original_shape[2] // Q  # 空间宽度 Patch 数
            x = self.proj(x)                      # (B, o*D, h, w)
            x = x.view(B, o, self.embed_dim, h, w) # (B, o, D, h, w)
            x = x.permute(0, 1, 3, 4, 2)      # (B, o, h, w, D)
            return x
        elif self.act=='inverse':
            (B, o, h, w, D) = x.shape
            C, H, W = self.input_original_shape
            c=C//o
            P=H//h
            Q=W//w
            x=x.permute(0,2,3,1,4)#(b,h,w,o,d)
            x = x.reshape(B, h * w, o,D)
            #k=min(o*D, P*Q*C) // 4 if linear_reduction=='auto' else linear_reduction#k = min(o*D, P*Q*C) // 4  # 恢复维度
            x = self.proj(x)  
            # (B, h*w, P*Q*C)
            x = x.reshape(B, h, w, o,P, Q, c)
            x = x.permute(0,3,6,1,4,2,5)  # (B, o,c, h, P, w, Q)
            x = x.reshape(B, C, H, W)  # 假设 h*P = H, w*Q = W ,C= o*c
            return x 





if __name__ == "__main__":
    x = torch.randn(8, 256, 64, 64)
    c = 16   # 每个 Patch 的通道数
    P = 8   # 空间 Patch 大小
    D = 800  # 嵌入维度
    
    patch_embed_1d=Patch_embedding_1d(D,(256,64,64),'normal')
    patch_unembed_1d=Patch_embedding_1d(D,(256,64,64),'inverse')
    patch_embed_2d=Patch_embedding_2d((P,P),D,(256,64,64),'normal')
    patch_unembed_2d=Patch_embedding_2d((P,P),D,(256,64,64),'inverse')
    patch_embed_3d=Patch_embedding_3d((c,P,P),D,(256,64,64),'normal')
    patch_unembed_3d=Patch_embedding_3d((c,P,P),D,(256,64,64),'inverse')
    print(patch_embed_1d(x).shape,patch_unembed_1d(patch_embed_1d(x)).shape)
    print(patch_embed_2d(x).shape,patch_unembed_2d(patch_embed_2d(x)).shape)
    print(patch_embed_3d(x).shape,patch_unembed_3d(patch_embed_3d(x)).shape)
    