# Get Modules 
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim

import torchvision 
import torchvision.transforms as transform 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# GPU Check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

# Define Model Architecture 
class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()

    # OUT SHAPE = ( IN - FILTER + 2*PADDING  / STRIDE  ) + 1    
    # (Conv layer x 3)  + (Linear layer x 2) 
    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)  # 32 x 32 x 64
    )
    
    self.layer2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)  # 64 x 16 x 32
    )

    self.layer3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)  # 128 x 8 x 16
    )

    self.linear = nn.Sequential(
        nn.Linear(128*8*16, 300, bias = True),
        nn.ReLU()
    )
    self.linear2 = nn.Linear(300, 3, bias = True)
    nn.init.xavier_uniform_(self.linear2.weight)

  def forward(self, X):
    out = self.layer1(X)
    out = self.layer2(out)
    out = self.layer3(out)

    # Flatten for linear layer 
    out = out.view(out.size(0), -1)

    out = self.linear(out)
    out = self.linear2(out)

    return out

# Set Model Parameters 
training_epochs = 10
learning_rate = 0.001
batch_size = 100

# Create Object 
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

# Set Dataset for Training 
trans = transform.Compose([transform.Resize((64, 128)), transform.ToTensor()])
dataset = ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/인공지능 응용/traing_dataset', transform=trans) 
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers=2)

# Train ... 
for epoch in range(training_epochs):
  avg_cost = 0  
  for X, y in dataloader:
    X = X.to(device)
    y = y.to(device)

    hypothesis = model(X)
    cost = criterion(hypothesis, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_cost += cost.item() / len(hypothesis)

    if epoch % 5 == 0:
      prediction = torch.argmax(hypothesis, axis = 1)
      correct = (prediction == y).sum()  # 소수점 x 
      ACC = correct/ len(hypothesis)

      print(f"Epoch : {epoch} || Accuracy : {ACC*100} || Correction : {correct} || Cost : {cost.item()} ")

print("Traninig finished")

# Set Test dataset 
trans = transform.Compose([transform.Resize((64, 128)), transform.ToTensor()])
dataset = ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/인공지능 응용/test_dataset', transform=trans) 
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers=4)

# Condition --- Learning X
with torch.no_grad(): 

  for X, y in dataloader:
    X = X.to(device)
    y = y.to(device)

    hypothesis = model(X)

    prediction = torch.argmax(hypothesis, axis = 1)
    correct = (prediction == y).sum() 
    ACC = correct/ len(hypothesis)

    print(f"Test Accuracy : {ACC*100} || Test Correction : {correct} ")

  print("Test Finished")

# Check with one Sample and Visualize 
with torch.no_grad():
  r = random.randint(0, len(dataset) - 1)

  sample = dataset.__getitem__(r)[0].unsqueeze(0).to(device)
  label = dataset.__getitem__(r)[1]

  hypothesis = model(sample)
  prediction = torch.argmax(hypothesis , axis = 1)
  corr = prediction == label

  print(f" # ----------- Prediction : {prediction.item()} || Correct : {corr.item()} --------------- #")

  plt.figure(figsize = (8,16))  
  plt.imshow(np.transpose(dataset.__getitem__(r)[0],(1,2,0)), cmap = 'Greys', interpolation = 'nearest')
  plt.show()  
    
