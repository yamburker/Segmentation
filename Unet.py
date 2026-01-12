import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,in_ch = 1,num_classes = 5):
        super().__init__()

        self.down1 = DoubleConv(in_ch,64)
        self.down2 = DoubleConv(64,128)
        self.down3 = DoubleConv(128,256)
        self.down4 = DoubleConv(256,512)

        self.pool = nn.MaxPool2d(2)

        self.up3 = DoubleConv(512 + 256,256)
        self.up2 = DoubleConv(256 + 128,128)
        self.up1 = DoubleConv(128 + 64,64)

        self.final = nn.Conv2d(64,num_classes,1)

    def forward(self, x):
        # Downsampling
        c1 = self.down1(x)  # 64
        c2 = self.down2(self.pool(c1))  # 128
        c3 = self.down3(self.pool(c2))  # 256
        c4 = self.down4(self.pool(c3))  # 512

        # Upsampling
        u3 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = self.up3(torch.cat([u3, c3], dim=1))  # 512+256 -> 256

        u2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, c2], dim=1))  # 256+128 -> 128

        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, c1], dim=1))  # 128+64 -> 64

        return self.final(u1)




