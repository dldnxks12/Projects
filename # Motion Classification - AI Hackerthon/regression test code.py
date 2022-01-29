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

# Load data - 증강된 data 가져올 것
train_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Hackerton/Dacon/train_features.csv")
train_label = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Hackerton/Dacon/train_labels.csv")

out_list = []
for i in range(3125):
    id = train_data.loc[(train_data['id'] == i)].values[:, 2:]
    id = np.reshape(id, (-1, 600 * 6))
    out_list.append(id)

out_list = np.array(out_list)
data = torch.from_numpy(out_list)
data = data.squeeze(1)

out_list = []
label_dict = {}
for i in range(3125):
    label = train_label.loc[(train_label['id'] == i)].values[0, 1]
    if label == 26:
      label = 1
    elif label != 26:
      label_dict[i] = label
      label = 0
    out_list.append(label)

out_list = np.array(out_list)
label = torch.from_numpy(out_list)

data , test = data[:2500] , data[2500:]
label, test_label = label[:2500] , label[2500:]

class Classify(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3600, 1800) # 4200 , 1800
        self.linear2 = nn.Linear(1800, 300)
        self.linear3 = nn.Linear(300, 100)
        self.linear4 = nn.Linear(100, 20)
        self.linear5 = nn.Linear(20, 1) # 0 1


    def forward(self, x):
        out = F.relu(self.linear(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = F.relu(self.linear4(out))
        out = F.relu(self.linear5(out))

        print("out.shape 1", out.shape)

        out = torch.sigmoid(out) # 20 개 x 1

        print("out.shape 2", out.shape)

        return out

from torch.utils.data import TensorDataset

model = Classify()

optimizer = optim.SGD(model.parameters(), lr=0.01)
batch_size = 20
num_epochs = 50

train_dataset = TensorDataset(data, label)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(num_epochs + 1):
    avg_cost = 0
    batch_length = len(train_loader)
    for x, y in train_loader:

        print(x.shape)
        print(y.shape)
        pred = model(x.float())
        print("pred 1", pred.shape)
        pred = pred.squeeze(1)
        print("pred 2", pred.shape)
        y = y.float()

        exit(0)

        cost = F.binary_cross_entropy(pred, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 10 == 0:
            prediction = pred >= torch.FloatTensor([0.5])  # 0.5 이상이면 True로
            correct_prediction = prediction.float() == y  # 실제 값과 일치하는 경우에 True
            acc = correct_prediction.sum().item() / len(correct_prediction)

            print(f" Cost : {cost.item()}, Acc : {acc * 100}")

    print(f"Epoch : {epoch}")

with torch.no_grad():

  test = test.float()
  test_label = test_label.float()

  prediction = model(test)
  prediction = prediction.squeeze(1)
  prediction = prediction >= torch.FloatTensor([0.9]) # 0.5 이상이면 True로
  correct_prediction = prediction == test_label # 실제 값과 일치하는 경우에 True
  acc = correct_prediction.sum().item() / len(correct_prediction)

print("check 1", test_label[40:50])
print("check 2", prediction[40:50])
print(acc)