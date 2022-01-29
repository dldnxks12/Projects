import torch
import torch.nn as nn
import torch.optim as optim # for optimizer 
import torch.nn.functional as F # for Softmax function

import torchvision.transforms as transforms # for tensor transforming
from torch.utils.data import TensorDataset  # for make dataset type
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# Upsample ? Pooling으로 인해서 작아진 것을 복원 with pytorch의 upsample 연산으로 구함 
# SegNet 논문에서는 Pooling하는 과정에서 Index를 기억해두었다가 Unpooling 시 이를 이용해서 UnPooling을 진행 
# Pytorch에서 제공하는 UpSampling (with bilinear)을 이용해서 수행!

# Paper architecture 

# Encoder 
# conv - conv - pool 
# conv - conv - pool
# conv - conv - conv - pool
# conv - conv - conv - pool
# conv - conv - conv - pool

# Decoder 
# upsample - conv - conv - conv
# upsample - conv - conv - conv
# upsample - conv - conv - conv
# upsample - conv - conv
# upsample - conv - conv - softmax 

# DIY architecture 

# Encoder 
# conv - conv - pool 
# conv - conv - pool
# conv - conv - pool

# Decoder 
# upsample - conv - conv 
# upsample - conv - conv 
# upsample - conv - conv - softmax 

class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_channel = 3
        in_height  = 224
        in_width   = 224
        
        num_class  = 1
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 32, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(32, 64, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 128, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(128, 256, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 512, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 1024, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),                                    
        )
        
        self.decoder = nn.Sequential(        
            nn.Upsample(scale_factor=2, mode ='bilinear', align_corners = True),
            nn.Conv2d(1024, 512, kernel_size =3, stride =1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 256, kernel_size =3, stride =1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 128, kernel_size =3, stride =1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            
            nn.Upsample(scale_factor=2, mode ='bilinear', align_corners = True),            
            nn.Conv2d(128, 64, kernel_size =3, stride =1, padding = 1),            
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 32, kernel_size =3, stride =1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            
            nn.Upsample(scale_factor=2, mode ='bilinear', align_corners = True),            
            nn.Conv2d(32, 16, kernel_size =3, stride =1, padding = 1),          
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(16, num_class, kernel_size =3, stride =1, padding = 1)                      
        )
        
    def forward(self, x): # input Image : Batchsize x 3 x 224 x 224        
        out = self.encoder(x)
        out = self.decoder(out)
        
        return out
    
model = SegNet()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

x_data = np.load('./cavana_x.npy')
y_data = np.load('./cavana_y.npy')

x_data = torch.tensor(x_data).float()
y_data = torch.tensor(y_data).float()
y_data = y_data.unsqueeze(1)

x_test = x_data[0]
y_test = y_data[0]
x_test2 = x_data[1]
y_test2 = y_data[1]
x_test3 = x_data[2]
y_test3 = y_data[2]

x_test = x_test.unsqueeze(0)
y_test = y_test.unsqueeze(0)
x_test2 = x_test2.unsqueeze(0)
y_test2 = y_test2.unsqueeze(0)
x_test3 = x_test3.unsqueeze(0)
y_test3 = y_test3.unsqueeze(0)


print(x_test.shape)
print(y_test.shape)

x_data = x_data[3:]
y_data = y_data[3:]

train_dataset = TensorDataset(x_data, y_data)

# DataLoader 
train_loader = DataLoader( dataset = train_dataset, batch_size = 100, shuffle = True, drop_last = True )

# train
for epoch in range(1):    
    avg_cost = 0
    correct = 0
    batch_length = len(train_loader)
    for x, y in train_loader:
        optimizer.zero_grad()
        
        pred = model(x)                
        cost = criterion(pred , y)        
        cost.backward()        
        optimizer.step()

        pred = (pred > 0.5).float()
        correct += (pred == y).sum()
        num_pixel = torch.numel(pred)
        
        avg_cost += cost / batch_length        
        print(f"epoch {epoch} Loop cost {cost} Correct{correct/num_pixel}")       
    print("Avg_cost: ", avg_cost)        
    
def color_map(image, nc = 1):
    
    label_colors = np.array([(0, 0, 0), (255, 255, 255)]) 
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for I in range(0, nc + 1):
        idx = image == 1
        
        r[idx] = label_colors[I, 0]
        g[idx] = label_colors[I, 1]
        b[idx] = label_colors[I, 2]
    
    rgb = np.stack([r, g, b], axis = 2)
    
    return rgb

with torch.no_grad():
    prediction = model(x_test3)
    prediction = torch.sigmoid(prediction)
    prediction = prediction.squeeze()    
    
prediction[prediction > 0.5 ]   = 1
prediction[prediction <= 0.5 ]  = 0

print(prediction)
print(prediction.shape)

rgb_pred = color_map(prediction)
print(rgb_pred.shape)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(prediction)
plt.subplot(1,2,2)
plt.imshow(y_test3[0][0])
