import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
""" class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16,kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        scale = self.sigmoid(out).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale


class SFEM(nn.Module):
    def __init__(self,inchannels,hidden_channels,conv2d_outchannels=200, conv3d_channels=16,ch_att_re=16,sa_kernel_size=7):
        super(SFEM, self).__init__()
        
        # 两个3D卷积层
        self.conv3d_1 = nn.Conv3d(1, conv3d_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv3d_2 = nn.Conv3d(conv3d_channels, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
        # 2D卷积层
        self.conv2d_1 = nn.Conv2d(inchannels, conv2d_outchannels, kernel_size=3, padding=1)
        
        # CBAM模块
        self.cbam = CBAM(conv2d_outchannels,reduction_ratio=ch_att_re,kernel_size=sa_kernel_size)
        
        # 2D卷积层
        self.conv2d_2 = nn.Conv2d(conv2d_outchannels, inchannels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    
        self.conv2d_3 = nn.Conv2d(inchannels,hidden_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 输入x的形状: (B, C, H, W)
        b, c, h, w = x.size()
        residual=x
        x=x.unsqueeze(1)
        x=self.conv3d_1(x)
        x=self.conv3d_2(x)
        x=x.squeeze(1)  #(B,C,H,W)
        
        # 第一个2D卷积
        x = self.relu(self.conv2d_1(x))  # (B,C, H, W)
        
        
        # CBAM模块
        x_cbam = self.cbam(x)  # (B, C, H, W)
        
        
        # 最后的2D卷积
        out = self.relu(self.conv2d_2(x_cbam))  # (B,C, H, W)
        
        out=out+residual
        out=self.conv2d_3(out)
        return out """
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.BatchNorm1d(channels // reduction_ratio),  # 添加BatchNorm
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.BatchNorm1d(channels)  # 添加BatchNorm
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        scale = self.sigmoid(out).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)  # 添加BatchNorm
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale
class ResnetBlock(nn.Module):
    def __init__(self, num_channel, kernel=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(num_channel)
        self.activation = nn.LeakyReLU(inplace=True)
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self, x):
        residual = x
        x = self.bn(self.conv1(x))
        x = self.activation(x)
        x = self.bn(self.conv2(x))
        x = torch.add(x, residual)
        return x

class SFEM(nn.Module):
    def __init__(self, inchannels, hidden_channels,num_residuals, conv2d_outchannels=200, conv3d_channels=16, ch_att_re=16, sa_kernel_size=7):
        super(SFEM, self).__init__()
        
        """ # 3D卷积部分
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, conv3d_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(conv3d_channels)
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(conv3d_channels, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(1)
        ) """
        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(conv2d_outchannels, kernel=3, stride=1, padding=1))
        self.residual_layers = nn.Sequential(*resnet_blocks)
        # 2D卷积部分
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(inchannels, conv2d_outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2d_outchannels)
        )
        
        # CBAM模块
        self.cbam = CBAM(conv2d_outchannels, reduction_ratio=ch_att_re, kernel_size=sa_kernel_size)
        
        # 2D卷积部分
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(conv2d_outchannels, inchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannels)
        )
        
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(inchannels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels)
        )
        
        self.relu = nn.LeakyReLU(inplace=True)
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                nn.init.zeros_(self._modules[m].bias)
    
    def forward(self, x):
        # 输入x的形状: (B, C, H, W)
        b, c, h, w = x.size()
        residual = x
        
        # 第一个2D卷积
        x = self.relu(self.conv2d_1(x))  # (B, C, H, W)
        residual1=x
        for _ in range(9):
            x = self.residual_layers(x)
            x = torch.add(x, residual1)
            
        # CBAM模块
        x_cbam = self.cbam(x)  # (B, C, H, W)
        
        # 最后的2D卷积
        out = self.relu(self.conv2d_2(x_cbam))  # (B, C, H, W)
        
        # 残差连接
        out = out + residual
        out = self.conv2d_3(out)
        return out
# 测试代码
if __name__ == "__main__":
    # 模拟输入: batch_size=4, channels=30, height=64, width=64
    x = torch.randn(4, 30, 64, 64)
    model = SFEM(30,100,3)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")  # 应该输出: torch.Size([4,100, 64, 64])