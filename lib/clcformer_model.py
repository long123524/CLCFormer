import torch
import torch.nn as nn
from config import swin_tiny_patch4_224 as swin
import torch.nn.functional as F
import math
from DFConv import DeformConv2d
import timm


class cSE(nn.Module):  # noqa: N801
    """
    The channel-wise SE (Squeeze and Excitation) block from the
    `Squeeze-and-Excitation Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x

class sSE(nn.Module):  # noqa: N801
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class scSE(nn.Module):  # noqa: N801
    """
    The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
    block from the `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class DWconv(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,padding=1,dilation=1):
        super(DWconv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class BiFFM(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFFM, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.residual = ATR(ch_1 + ch_2, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dw1 = DWconv(ch_2,ch_2 // r_2,padding=8,dilation=8)
        self.dw2 = DWconv(ch_2, ch_2 // r_2,padding=8,dilation=8)
        self.dw3 = DWconv(ch_2 // r_2, ch_2, padding=8, dilation=8)
        self.df_conv = DeformConv2d(ch_1,ch_int)
        self.scse = scSE(ch_int)
        self.dw4 = DWconv(ch_int, ch_out, padding=8, dilation=8)

    def forward(self, g, x):
        ##Transformer_branch

        y1 = self.avg_pool(x)
        y1 = self.dw1(y1)
        y2 = self.max_pool(x)
        y2 = self.dw2(y2)
        y = self.relu(y1+y2)
        y = self.dw3(y)
        y = self.sigmoid(y)*x

        ##CNN_branch
        c1 = self.df_conv(g)
        c1 = self.scse(c1)
        c1 = self.dw4(c1)
        c2 = self.sigmoid(c1)*g

        fuse = self.residual(torch.cat([y, c2], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse



class CLCFormer(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.4, normal_init=True, pretrained=False):
        super(CLCFormer, self).__init__()

        self.efficienet = timm.create_model('efficientnet_b3')
        if pretrained:
            self.efficienet.load_state_dict(torch.load('./pretrained/efficientnet_b3_ra2-cf984f9c.pth'))

        self.transformer = swin(pretrained=pretrained)


        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)

        ###
        self.up3 = Up(64, 32)

        self.final_x = nn.Sequential(
            Conv(232, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_3 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFFM(ch_1=232, ch_2=768, r_2=2, ch_int=256, ch_out=232, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFFM(ch_1=136, ch_2=384, r_2=2, ch_int=128, ch_out=136,
                                       drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=232, out_ch=128, in_ch2=136, attn=True)

        self.up_c_2_1 = BiFFM(ch_1=48, ch_2=192, r_2=1, ch_int=64, ch_out=48, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(128, 64, 48, attn=True)

        ###
        self.up_c_3_1 = BiFFM(ch_1=32, ch_2=96, r_2=1, ch_int=32, ch_out=32, drop_rate=drop_rate / 2)
        self.up_c_3_2 = Up(64, 32, 32, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs):
        # transformer path
        x_b, out4 = self.transformer(imgs)
        x_b_1 = x_b[0]
        x_b_1 = torch.transpose(x_b_1, 1, 2)
        x_b_1 = x_b_1.view(x_b_1.shape[0], -1, 128, 128)
        x_b_1 = self.drop(x_b_1)

        x_b_2 = x_b[1]
        x_b_2 = torch.transpose(x_b_2, 1, 2)
        x_b_2 = x_b_2.view(x_b_2.shape[0], -1, 64, 64)
        x_b_2 = self.drop(x_b_2)

        x_b_3 = x_b[2]
        x_b_3 = torch.transpose(x_b_3, 1, 2)
        x_b_3 = x_b_3.view(x_b_3.shape[0], -1, 32, 32)
        x_b_3 = self.drop(x_b_3)

        x_b_4 = out4
        x_b_4 = torch.transpose(x_b_4, 1, 2)
        x_b_4 = x_b_4.view(x_b_4.shape[0], -1, 16, 16)
        x_b_4 = self.drop(x_b_4)

        # CNN path
        ####effinetb3
        x_u128 = self.efficienet.conv_stem(imgs)
        x_u128 = self.efficienet.bn1(x_u128)
        x_u128 = self.efficienet.act1(x_u128)        ##
        x_u128 = self.efficienet.blocks[0](x_u128)
        x_u64 = self.efficienet.blocks[1](x_u128)

        x_u_2 = self.efficienet.blocks[2](x_u64)
        x_u_2 = self.drop(x_u_2)

        x_u_3 = self.efficienet.blocks[3](x_u_2)
        x_u_3 = self.drop(x_u_3)

        x_u_3 = self.efficienet.blocks[4](x_u_3)
        x_u_3 = self.drop(x_u_3)

        x_u = self.efficienet.blocks[5](x_u_3)
        x_u = self.drop(x_u)

        # joint path
        x_c = self.up_c(x_u, x_b_4)

        x_c_1_1 = self.up_c_1_1(x_u_3, x_b_3)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)

        ###
        x_c_3_1 = self.up_c_3_1(x_u64, x_b_1)
        x_c_3 = self.up_c_3_2(x_c_2, x_c_3_1)

        #
        map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear')
        map_1 = F.interpolate(self.final_1(x_c_1), scale_factor=16, mode='bilinear')
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=8, mode='bilinear')
        map_3 = F.interpolate(self.final_3(x_c_3), scale_factor=4, mode='bilinear')
        return map_x, map_1, map_2, map_3

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.up3.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final_3.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)
        self.up_c_3_1.apply(init_weights)
        self.up_c_3_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ATR(in_ch1+in_ch2, out_ch)

        if attn:
            self.attn_block = ATG(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

####
class ATG(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(ATG,self).__init__()
        self.W_g = nn.Sequential(
            DWconv(F_g, F_int, stride=1,padding=4,dilation=4),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            DWconv(F_l, F_int, stride=1,padding=6,dilation=6),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            DWconv(F_int, 1, stride=1,padding=8,dilation=8),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi1 = self.psi(psi)*x
        return psi1


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))


class ATR(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(ATR, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        self.se = cSE(output_dim)

    def forward(self, x):

        return self.se(self.conv_block(x) + self.conv_skip(x))


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
