# Classification

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

print("Device", device)

# Load data
train_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Hackerton/Dacon/train_features.csv")
train_label = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Hackerton/Dacon/train_labels.csv")

# sensor data to tensor
out_list = []
for i in range(3125):
    id = train_data.loc[(train_data['id'] == i)].values[:, 2:]
    out_list.append(id)

out_list = np.expand_dims(np.array(out_list), axis=1)
data = torch.from_numpy(out_list)

# label data to tensor
out_list = []
for i in range(3125):
    label = train_label.loc[(train_label['id'] == i)].values[0, 1]
    out_list.append(label)

out_list = np.array(out_list)
# out_list = np.expand_dims(np.array(out_list), axis = 1)
label = torch.from_numpy(out_list)


class VGG_ORG(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.layer1 = nn.Sequential(  # 1 x 600 x 6
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(  # 32 x 300 x 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.fc1 = nn.Linear(64 * 298 * 1, 100)
        self.fc2 = nn.Linear(100, 61)

        # self.last = nn.Softmax()

    def forward(self, x):
        # Conv layer
        out = self.layer1(x)  # out shape torch.Size([100, 32, 300, 3])
        out = self.layer2(out)  # out shape torch.Size([100, 64, 298, 1])

        # flatten
        out = out.view(-1, 64 * 298)

        # fc layer
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out
