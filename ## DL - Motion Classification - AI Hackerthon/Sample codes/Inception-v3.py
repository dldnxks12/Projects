# Classification

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

print("Device", device)

# Load data
train_data = pd.read_csv("C:/Users/USER/Desktop/Hackerton/train_features.csv")
train_label = pd.read_csv("C:/Users/USER/Desktop/Hackerton/train_labels.csv")
test_data = pd.read_csv("C:/Users/USER/Desktop/Hackerton/test_features.csv")

out_list = []
for i in range(3125):
  id = train_data.loc[(train_data['id'] == i)].values[:,2:]
  out_list.append(id)

out_list = np.expand_dims(np.array(out_list), axis = 1)
data =  out_list

# Scaling
act_list=train_data.iloc[:,2:].columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[act_list]= scaler.fit_transform(train_data[act_list])

out_list = []
for i in range(3125):
  id = train_data.loc[(train_data['id'] == i)].values[:,2:]
  out_list.append(id)

temp_data = np.expand_dims(np.array(out_list), axis = 1)

li = [temp_data, data]
data = np.concatenate(li, axis = 0)
data = torch.from_numpy(data)

out_list = []
for i in range(3125):
  label = train_label.loc[(train_label['id'] == i)].values[0,1]
  out_list.append(label)

out_list = np.array(out_list)

li = [out_list, out_list]
label = np.concatenate(li, axis = 0)
label = torch.from_numpy(label)

data, valid_data = data[:5000], data[5000:]
label, valid_label = label[:5000], label[5000:]

# test data processing
out_list = []
for i in range(782):
  id = test_data.loc[(test_data['id'] == i + 3125)].values[:,2:]
  out_list.append(id)

out_list = np.expand_dims(np.array(out_list), axis = 1)
test_data = torch.from_numpy(out_list)

class Inception(nn.Module):
  def __init__(self, in_channel):
    super().__init__()
    # Block 1
    self.branch1_1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1)
    self.branch1_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    # Block 2
    self.branch3_1 = nn.Conv2d(in_channel, 16, kernel_size=1)  # 1x1 Conv
    self.branch3_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.branch3_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

    # Block 3
    self.branch_pool = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    # print("Input shape : ", np.shape(x)) # torch.Size([100, 8, 301, 4])

    branch1x1 = self.branch1_1(x)
    branch1x1 = self.branch1_2(branch1x1)

    branch3x3 = self.branch3_1(x)
    branch3x3 = self.branch3_2(branch3x3)
    branch3x3 = self.branch3_3(branch3x3)

    branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = self.branch_pool(branch_pool)

    # 3개의 output들을 1개의 list로
    outputs = [branch1x1, branch3x3, branch_pool]  # np.shape(outputs)) (3,)

    # torch.cat (concatenate)
    cat = torch.cat(outputs, 1)  # outputs list의 tensor들을 dim = 1로 이어준다.

    #cat.shape : torch.Size([300, 32, 301, 4])
    return cat


class Classification(nn.Module):
  def __init__(self):
    super().__init__()

    self.Conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
    self.Conv2 = nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=1)

    self.Incept1 = Inception(in_channel=8)
    self.Incept2 = Inception(in_channel=16)

    self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.mp2 = nn.MaxPool2d(kernel_size=3, stride=1)

    self.fc1 = nn.Linear(96 * 298 * 1, 3000)
    self.fc2 = nn.Linear(3000, 1000)
    self.fc3 = nn.Linear(1000, 61)

  def forward(self, x):

    out = self.Conv1(x)  # out_channel = 8
    out = F.relu(self.mp1(out))
    out = self.Incept1(out)  # out_channel = 96

    out = self.Conv2(out)  # out_channel = 16
    out = F.relu(self.mp2(out))
    out = self.Incept2(out)  # out_channel = 96

    out = out.view(-1, 96 * 298 * 1)

    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = F.relu(self.fc3(out))

    return out

model = Classification().to(device)

batch_size = 200
learning_rate = 0.005
num_epochs = 200  # 87

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

from torch.utils.data import TensorDataset

train_dataset = TensorDataset(data, label)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(num_epochs + 1):
  avg_cost = 0
  batch_length = len(train_loader)
  for x, y, in train_loader:

    y = y.long().to(device)

    pred = model(x.float().to(device))  # 100 x 61

    cost = loss(pred, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    corr = torch.argmax(pred)
    num_correct = (corr == y).sum().item()
    avg_cost += cost / batch_length
    acc = num_correct / batch_length

  print(f"Epoch : {epoch}")
  print(f"Correct {num_correct}/{batch_length}")
  print("Average Cost", avg_cost)

'''
with torch.no_grad():  # Gradient 학습 x

  valid_data = valid_data.float().to(device)
  valid_label = valid_label.long().to(device)

  prediction = model(valid_data)
  correct_prediction = torch.argmax(prediction, 1) == valid_label
  accuracy = correct_prediction.float().mean()
  print('Accuracy:', accuracy.item())

print("check 1", valid_label[:30])
print("check 2", torch.argmax(prediction, 1)[:30])
'''

submission = pd.read_csv('C:/Users/USER/Desktop/Hackerton/sample_submission.csv')

with torch.no_grad():  # Gradient 학습 x

  test_data = test_data.float().to(device)

  prediction = model(test_data)
  prediction = F.softmax(prediction)
  print(prediction.shape)

prediction = prediction.detach().cpu().numpy()
submission.iloc[:, 1:] = prediction
submission.to_csv('js_submission_1.csv', index=False)

