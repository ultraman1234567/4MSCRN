import torch
import torch.nn as nn
class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None):
        super().__init__()

        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.add_module("pw", torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))

        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        ))
class depthwise_separable_Conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_Conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class PAConv(nn.Module):
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        # self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        #self.k3 = BSConvU(nf, nf, kernel_size=k_size, padding =(k_size - 1) // 2)
        # self.k3 = ScConv(nf, nf, group_num=6, group_kernel_size=k_size)
        self.k3 = depthwise_separable_Conv(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.k4 = depthwise_separable_Conv(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        #self.k4 = BSConvU(nf, nf, kernel_size=k_size, padding =(k_size - 1) // 2)


    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        return out
# 多尺度自校准卷积模块
class SCPA(nn.Module):
    def __init__(self, hidden_dim, reduction=3, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = hidden_dim // reduction
        self.conv1_a = nn.Conv2d(hidden_dim, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(hidden_dim, group_width, kernel_size=1, bias=False)
        self.conv1_c = nn.Conv2d(hidden_dim, group_width, kernel_size=1, bias=False)
        """ self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False),
        ) """
        self.k1 = depthwise_separable_Conv(group_width, group_width, kernel_size=3, padding=1, bias=False)
        self.PAConv = PAConv(group_width)
        self.PAConv2 = PAConv(group_width, 5)
        self.conv3 = nn.Conv2d(
            group_width * reduction, hidden_dim, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # self.gelu = nn.GELU()
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(self._modules[m].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(self._modules[m].bias)
    def forward(self, x):
        residual = x
        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_c = self.conv1_c(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out_c = self.lrelu(out_c)
        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_c = self.PAConv2(out_c)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out_c = self.lrelu(out_c)
        out = self.conv3(torch.cat([out_a, out_b,out_c], dim=1))
        #out += residual
        return out
    
if __name__ == "__main__":
    x=torch.randn(1,254,64,64).to('cuda')
    scpa=SCPA(254).cuda()
    print(scpa(x).shape)
