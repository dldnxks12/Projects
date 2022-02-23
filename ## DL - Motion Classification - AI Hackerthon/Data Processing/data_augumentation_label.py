
import pandas as pd
import numpy as np


train_labels = pd.read_csv("C:/Users/USER/Desktop/Hackerton/train_labels.csv")

# train label processing
n = 4
train_labels_copy = train_labels.copy()
for i in range(0, n):
    train_labels = train_labels.append(train_labels_copy, ignore_index=True)
train_labels = train_labels.reset_index()

print(np.shape(train_labels))

train_labels.to_csv('js_train_label.csv', index=False)






















