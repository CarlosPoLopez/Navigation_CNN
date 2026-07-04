import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # 3x3 weight matrix (kernel) that slides over the images
            nn.BatchNorm2d(out_ch),  # subtracts the mean and divides by the standard deviation
            nn.ReLU(inplace=True),   # activation: 0 if <0, unchanged if >0
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    # Declares the layers (U-Net architecture)
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(ch, ch * 2)
        ch = ch * 2

        for f in reversed(features):
            self.upsamples.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))
            ch = f

        self.head = nn.Conv2d(ch, out_channels, 1)

    # Defines the computation (how the tensor flows through the network)
    def forward(self, x):
        # Encoder (lowers the resolution and increases channels = more information)
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder (raises the resolution and reduces channels)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)
