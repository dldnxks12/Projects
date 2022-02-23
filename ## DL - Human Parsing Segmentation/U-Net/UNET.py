# 편의를 위해 padding을 사용하였음
# Paper에서는 padding을 사용하지 않아 크기가 점점 줄어들고, Up-sampling과정에서 이전 layer의 heat map을 연결하는 과정에서 Cropped Concatenate이 필요함

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias = False), # batch norm을 수행함에 따라 bias는 내부적으로 사라진다 - beta value가 이 역할을 대신할 것
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace =True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias = False), # 한 Step 당 두 번의 Conv를 수행하고 두번 째 Conv는 channel을 유지
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512],
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride =2)

        # Down Part , features : 64 - 128 - 256 - 512
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up Part , reversed features : 512 - 256 - 128 - 64
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size = 2, stride = 2)) # 2배씩 Up
            self.ups.append(DoubleConv(feature*2, feature)) # 왜 feature*2? --- Skip Connection으로 Concatenate해줄 것

        self.bottlenect = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)


    def forward(self, x):

        skip_connections = [] # skip connection을 위해 down sampling 과정에서 Conv layer output들을 append할 것

        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # 순서에 주의! Pooling 전에 Skip conenction에 담아야 한다.
            x = self.pool(x)

        x = self.bottlenect(x)
        skip_connections = skip_connections[::-1] # reversing list

        for idx in range(0 , len(self.ups), 2): # 짝수 Idx : Transpose Conv
            x = self.ups[idx](x) # Up sample
            skip_connection = skip_connections[idx//2] # bring Skip Connection data

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:]) # height, width matching

            concat_skip = torch.cat((skip_connection, x), dim = 1)  # channel에 대해 Concatenate (batch x channel x height x width )
            x = self.ups[idx+1](concat_skip) # Concatenate 된 layer DoubleConv 통과

        return self.final_conv(x)


def test():
    x = torch.randn( (3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)

    print(preds.shape) # torch.Size([3, 1, 160, 160])
    print(x.shape)     # torch.Size([3, 1, 160, 160])

    assert preds.shape == x.shape


def segmap(image, nc=1):
    label_colors = np.array([(0, 0, 0), (255, 255, 255)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc + 1):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

if __name__ == "__main__":
    test()









