import torch
from torch import nn
import torch.nn.functional as F
# from CBAM import CBAM
# from CorrdAtt import CoordAtt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="true"


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel=32, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                # nn.Conv3d(channel, channel // reduction, kernel_size=(1, 1, 1), padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                # nn.Conv3d(channel // reduction, channel, kernel_size=(1, 1, 1), padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):

    def __init__(self, channel=32):
        super(RCAB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            # nn.Conv3d(channel, channel, (3, 3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            # nn.Conv3d(channel, channel, (3, 3, 3), 1, 1),
        )
        self.ca = CALayer(channel)
        # self.ca = CBAM(channel)
        # self.ca = CoordAtt(channel, channel)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.ca(y1)
        return x + y2


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, B_RCAB, channel=32):
        super(ResidualGroup, self).__init__()
        module_body = []
        for i in range(B_RCAB):
            module_body.append(RCAB(channel))
        module_body.append(nn.Conv2d(channel, channel, 3, 1, 1))
        # module_body.append(nn.Conv3d(channel, channel, (3, 3, 3), 1, 1))
        self.body = nn.Sequential(*module_body)

    def forward(self, x):
        res = self.body(x)

        res += x
        return res


class DoubleConv1(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.):
        super(DoubleConv1, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),    
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout, inplace=False))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x) + x
        return out


def convt2x2(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)


def pixel_shuffle(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    new_ch = ch // (scale_factor)
    new_height = height * scale_factor
    input_view = tensor.contiguous().view(
        num, new_ch, scale_factor, 1, height, width)
    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return shuffle_out.view(num, new_ch, new_height, width)


class URCANsub(nn.Module):
    def __init__(self, c_in, c_out, channel=32, n_rcab=5):
        super(URCANsub, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        self.head = DoubleConv1(c_in, channel)
        self.rg1 = nn.Sequential(ResidualGroup(n_rcab, channel * 1), DoubleConv1(channel, channel * 2))
        self.rg2 = nn.Sequential(DoubleConv1(channel * 2, channel * 4))
        self.rg3 = nn.Sequential(DoubleConv1(channel * 4, channel * 8))
        self.rg4 = nn.Sequential(DoubleConv1(channel * 8, channel * 16))

        self.up1 = convt2x2(channel * 16, channel * 8)
        self.rg5 = nn.Sequential(DoubleConv1(channel * 16, channel * 8))

        self.up2 = convt2x2(channel * 8, channel * 4)
        self.rg6 = nn.Sequential(DoubleConv1(channel * 8, channel * 4))

        self.up3 = convt2x2(channel * 4, channel * 2)
        self.rg7 = nn.Sequential(DoubleConv1(channel * 4, channel * 2))

        self.up4 = convt2x2(channel * 2, channel * 1)
        self.rg8 = nn.Sequential(DoubleConv1(channel * 2, channel * 1), ResidualGroup(n_rcab, channel * 1))

        # self.rg1 = nn.Sequential(ResidualGroup(n_rcab, channel * 1), DoubleConv1(channel, channel * 2))
        # self.rg2 = nn.Sequential(DoubleConv1(channel * 2, channel * 4))
        # self.rg3 = nn.Sequential(ResidualGroup(n_rcab, channel * 4), DoubleConv1(channel * 4, channel * 8))
        # self.rg4 = nn.Sequential(DoubleConv1(channel * 8, channel * 16))
        #
        # self.up1 = convt2x2(channel * 16, channel * 8)
        # self.rg5 = nn.Sequential(DoubleConv1(channel * 16, channel * 8))
        #
        # self.up2 = convt2x2(channel * 8, channel * 4)
        # self.rg6 = nn.Sequential(DoubleConv1(channel * 8, channel * 4),  ResidualGroup(n_rcab, channel * 4))
        #
        # self.up3 = convt2x2(channel * 4, channel * 2)
        # self.rg7 = nn.Sequential(DoubleConv1(channel * 4, channel * 2))
        #
        # self.up4 = convt2x2(channel * 2, channel * 1)
        # self.rg8 = nn.Sequential(DoubleConv1(channel * 2, channel * 1), ResidualGroup(n_rcab, channel * 1))

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.conv1_upsample = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv2_upsample = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv_last = nn.Conv2d(channel, c_out, 3, 1, 1)  

        # self.tail = nn.Sequential(
        #     nn.Conv2d(channel // 4, channel, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel * 1, c_out, 3, 1, 1))

    def forward(self, x):
        out = self.head(x)
        skip1 = out
        out = self.maxpool(out)

        out = self.rg1(out)
        skip2 = out
        out = self.maxpool(out)

        out = self.rg2(out)
        skip3 = out
        out = self.maxpool(out)

        out = self.rg3(out)
        skip4 = out
        out = self.maxpool(out)

        out = self.rg4(out)
        out = self.up1(out)
        out = torch.cat((skip4, out), dim=1)

        out = self.rg5(out)
        out = self.up2(out)
        out = torch.cat((skip3, out), dim=1)

        out = self.rg6(out)
        out = self.up3(out)
        out = torch.cat((skip2, out), dim=1)

        out = self.rg7(out)
        out = self.up4(out)
        out = torch.cat((skip1, out), dim=1)

        out = self.rg8(out) + skip1
        out = self.conv_before_upsample(out)
        out = self.conv1_upsample(out)
        out = pixel_shuffle(out, scale_factor=1)
        out = self.conv2_upsample(out)
        out = pixel_shuffle(out, scale_factor=1)
        out = self.conv_last(out)
        # out = pixel_shuffle(out, scale_factor=4)
        # out = self.tail(out)
        return out


# class URCANsub(nn.Module):
#     def __init__(self, c_in, c_out, channel=32, n_rcab=20):
#         super(URCANsub, self).__init__()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.head = nn.Conv2d(c_in, channel, 3, 1, 1)
#         self.rg1 = nn.Sequential(ResidualGroup(n_rcab, channel * 1), nn.Conv2d(channel, channel * 2, 3, 1, 1))
#         self.rg2 = nn.Sequential(nn.Conv2d(channel * 2, channel * 4, 3, 1, 1))
#         self.rg3 = nn.Sequential(nn.Conv2d(channel * 4, channel * 8, 3, 1, 1))
#         self.rg4 = nn.Sequential(nn.Conv2d(channel * 8, channel * 16, 3, 1, 1))
#
#         self.up1 = convt2x2(channel * 16, channel * 8)
#         self.rg5 = nn.Sequential(nn.Conv2d(channel * 16, channel * 8, 3, 1, 1))
#
#         self.up2 = convt2x2(channel * 8, channel * 4)
#         self.rg6 = nn.Sequential(nn.Conv2d(channel * 8, channel * 4, 3, 1, 1))
#
#         self.up3 = convt2x2(channel * 4, channel * 2)
#         self.rg7 = nn.Sequential(nn.Conv2d(channel * 4, channel * 2, 3, 1, 1))
#
#         self.up4 = convt2x2(channel * 2, channel * 1)
#         self.rg8 = nn.Sequential(nn.Conv2d(channel * 2, channel * 1, 3, 1, 1), ResidualGroup(n_rcab, channel * 1))
#         self.conv_before_upsample = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1),
#                                                   nn.LeakyReLU(inplace=True))
#         self.conv1_upsample = nn.Conv2d(channel, channel * 2, 3, 1, 1)
#         self.conv2_upsample = nn.Conv2d(channel, channel * 2, 3, 1, 1)
#         # self.tail = nn.Sequential(
#         #     nn.Conv2d(channel // 4, channel, 3, 1, 1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(channel * 1, c_out, 3, 1, 1))
#
#     def forward(self, x):
#         out = self.head(x)
#         skip1 = out
#         out = self.maxpool(out)
#
#         out = self.rg1(out)
#         skip2 = out
#         out = self.maxpool(out)
#
#         out = self.rg2(out)
#         skip3 = out
#         out = self.maxpool(out)
#
#         out = self.rg3(out)
#         skip4 = out
#         out = self.maxpool(out)
#
#         out = self.rg4(out)
#         out = self.up1(out)
#         out = torch.cat((skip4, out), dim=1)
#
#         out = self.rg5(out)
#         out = self.up2(out)
#         out = torch.cat((skip3, out), dim=1)
#
#         out = self.rg6(out)
#         out = self.up3(out)
#         out = torch.cat((skip2, out), dim=1)
#
#         out = self.rg7(out)
#         out = self.up4(out)
#         out = torch.cat((skip1, out), dim=1)
#
#         out = self.rg8(out) + skip1
#         out = self.conv_before_upsample(out)
#         out = self.conv1_upsample(out)
#         out = pixel_shuffle(out, scale_factor=2)
#         out = self.conv2_upsample(out)
#         out = pixel_shuffle(out, scale_factor=2)
#         out = self.conv_last(out)
#         # out = pixel_shuffle(out, scale_factor=4)
#         # out = self.tail(out)
#         return out


# if __name__ == '__main__':
#     from torchinfo import summary
#     import torch
#     x = torch.rand(1, 1, 256, 256)
#     print(x.shape)
#     print(net(x).shape)
#     summary(net, input_size=(1, 1, 256, 256), col_names=("input_size", "output_size", "num_params", "mult_adds"),
#             depth=4)