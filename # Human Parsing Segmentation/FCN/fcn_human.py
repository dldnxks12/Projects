# Human Parsing dataset을 이용한 7. Segmentation-Semantic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.models.vgg import VGG # Pretrained VGG Model
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output
      
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)           
  

class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)  
      
ranges = {'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))}
cfg = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

vgg_model = VGGNet(requires_grad = True)
fcn = FCNs(pretrained_net = vgg_model, n_class = 2)

optimizer = optim.SGD(fcn.parameters(), lr = 0.01, momentum = 0.7)
criterion = nn.BCELoss()

# dataset load (total 6000개의 data 중 1000만 사용)
x_train = np.load('./x_train.npz')['data'][:1000].astype(np.float32)
y_train = np.load('./y_train.npz')['data'][:1000].astype(np.float32)

# Handling Dimension - change channel position idx 3 to 1
x_train = np.transpose(x_train, (0, 3, 1, 2))
y_train = np.transpose(y_train, (0, 3, 1, 2))

# Numpy to Tensor for pytorch tranining
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

# DataLoader에 넣어줄 dataset type 생성
train_dataset = TensorDataset(x_train, y_train)

# Make DataLoader 
train_loader = DataLoader( dataset = train_dataset, batch_size = 100, shuffle = True, drop_last = True )

# train ... 
avg_cost = 0
for epoch in range(2):    
    batch_length = len(train_loader)
    for x, y in train_loader:
        
        optimizer.zero_grad()
        
        output = fcn(x)
        output = F.sigmoid(output) # sigmoid 대신 softmax를 사용할 수 도 있다. How? fcn의 마지막 layer를 flatten 한 후 softmax를 통과시켜 다시 reshape해서 보내주면 된다.
        
        cost = criterion(output , y)
        
        cost.backward()        
        optimizer.step()
        
        avg_cost += cost / batch_length        
        print("Loop cost : ", cost)
        
    print("Avg_cost: ", avg_cost)     
    
    
# RGB Sememtation map     
def decode_segmap(image, nc=2):
  
    label_colors = np.array([(128, 0, 0), (0, 128, 0)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
    return rgb    
  
# test .... 

### test 

# test data load
x_test = np.load('./x_train.npz')['data'][-1].astype(np.float32)
y_test = np.load('./y_train.npz')['data'][-1].astype(np.float32)

# handling data
x_test = np.transpose(x_test, (2, 0, 1))
y_test = np.transpose(y_test, (2, 0, 1))
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

x_test = x_test.unsqueeze(0)
y_test = y_test.unsqueeze(0)

# make prediction 
prediction = fcn(x_test)
print(prediction.shape)

# 1. squeeze() - (1, 2, 224, 224) -> (224, 244) 
# 2. dim = 0에 대해 argmax (모든 channel 중 가장 큰 값을 갖는 Index를 반환)

pred = torch.argmax(prediction.squeeze(), dim = 0).detach().numpy() 
print(pred.shape)

# Index에 맞는 Color map으로 mapping
rgb_pred = decode_segmap(pred)
print(rgb_pred.shape)

# Prediction check
plt.subplot(1,2,1)
plt.imshow(rgb_pred[:,:,:])
plt.subplot(1,2,2)
plt.imshow(y_test[0][0])
plt.show()
