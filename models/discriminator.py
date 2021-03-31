import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.convolution_section = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            ConvolutionalBlock(64, 64, 2),
            ConvolutionalBlock(64, 128, 1),
            ConvolutionalBlock(128, 128, 2),
            ConvolutionalBlock(128, 256, 1),
            ConvolutionalBlock(256, 256, 2),
            ConvolutionalBlock(256, 512, 1),
            ConvolutionalBlock(512, 512, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        out = self.convolution_section(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

class ConvolutionalBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super(ConvolutionalBlock, self).__init__()
        # Must use Sequential so we dont overide batchnorms later
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        out = self.block(x)
        return out

if __name__ == '__main__':
    D = Discriminator()
    print(D)
