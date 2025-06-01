import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
class SubPixelConv(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixelConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 
                             kernel_size=3, padding=1),nn.BatchNorm2d(out_channels * (upscale_factor ** 2)))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class ProgressiveSubPixelUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, upscale_factor_per_step=2):
        super(ProgressiveSubPixelUpsampler, self).__init__()
        self.upsample_blocks = nn.ModuleList()
        num_upsamples=upscale_factor//upscale_factor_per_step
        current_channels = in_channels
        for i in range(num_upsamples):
            self.upsample_blocks.append(
                SubPixelConv(current_channels, 
                           out_channels if i == num_upsamples-1 else current_channels,
                           upscale_factor_per_step)
            )
            current_channels = current_channels
            
    def forward(self, x):
        for block in self.upsample_blocks:
            x = block(x)
        return x
if __name__ == "__main__":
    x=torch.randn(1,64,32,32)
    up=ProgressiveSubPixelUpsampler(64,64)
    print(up(x).shape)
    
import math
 
import torch
import torch.nn as nn
 
 
class Net(nn.Module):
    def __init__(self, in_channels, base_channel, out_channels,upscale_factor, num_residuals_pre=0,num_residuals_post=3):
        super(Net, self).__init__()
 
        self.input_conv = nn.Conv2d(in_channels, base_channel, kernel_size=3, stride=1, padding=1)
 
        resnet_blocks_pre = []
        for _ in range(num_residuals_pre):
            resnet_blocks_pre.append(ResnetBlock(base_channel, kernel=3, stride=1, padding=1))
        self.residual_layers_pre = nn.Sequential(*resnet_blocks_pre)
        resnet_blocks_post = []
        for _ in range(num_residuals_post):
            resnet_blocks_post.append(ResnetBlock(base_channel, kernel=3, stride=1, padding=1))
        self.residual_layers_post = nn.Sequential(*resnet_blocks_post)
        self.mid_conv = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)
 
        upscale = []
        for _ in range(int(math.log2(upscale_factor))):
            upscale.append(PixelShuffleBlock(base_channel, base_channel, upscale_factor=2))
        self.upscale_layers = nn.Sequential(*upscale)
 
        self.output_conv = nn.Conv2d(base_channel, out_channels, kernel_size=3, stride=1, padding=1)
 
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                nn.init.zeros_(self._modules[m].bias)
               
 
    def forward(self, x):
        x = self.input_conv(x)
        residual = x
        x = self.residual_layers_pre(x)
        x = self.mid_conv(x)
        x = torch.add(x, residual)
        x = self.upscale_layers(x)
        residual1=x
        for _ in range(9):
            x = self.residual_layers_post(x)
            x = torch.add(x, residual1)
        x = self.output_conv(x)
        return x
 
 
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
 
 
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
 
 
class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor, kernel=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2, kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)
 
    def forward(self, x):
        x = self.ps(self.conv(x))
        return x