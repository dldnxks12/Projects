'''

1. torchvision.datasets.ImageFolder
2. torchvision.transforms 

transpose.Compose([ transforms.Resize(224,224)]) # transforms.ToTensor()는 사용하지 않음 - data를 save해두기 위해서 우선 데이터 type그대로 유지

if dataloader에 넣어줄 dataset class를 구현하는 경우에는 데이터를 하나씩 가져와서 tensor로 바꾸어 주면 된다. --- 이 부분은 또 따로 구현해서 Code 올릴 것 

'''
import torchvision

import torchvision.transforms as transforms # image resize를 위한 transforms 
from torch.utils.data import DataLoader

# for visualization
import matplotlib.pyplot as plt
import numpy as np


tf = transforms.Compose([        
        transforms.Resize((224,224)),
])

train_data = torchvision.datasets.ImageFolder(root='./origin/', transform = tf)

x_data = []
y_data = []
# Image Load and save  - origin folder 안에 x label folder 만 있음 
for idx, value in enumerate(train_data):
    
    data, label = value      
    data = np.array(data) # numpy type으로 바꿔서 저장

    x_data.append(data)    

# Image Load and save  - origin folder 안에 y label folder 만 있음 
for idx, value in enumerate(train_data):
    
    data, label = value   
    data = np.array(data) # numpy type으로 바꿔서 저장

    y_data.append(data)
        
        
np.save("./numpy_x/" , x_data)            
np.save("./numpy_y/" , y_data)    
    

# numpy_x, numpy_y folder는 os module을 사용해서 미리 만들어두자
