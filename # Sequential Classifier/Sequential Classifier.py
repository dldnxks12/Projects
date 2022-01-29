import os
import torch
import torch.nn as nn
from torchtext.legacy import data, datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed(777)

TEXT = data.Field(sequential = True, batch_first = True, lower = True)
LABEL = data.Field(sequential = False, batch_first = True)

trainset, validset, testset = datasets.SST.splits(TEXT, LABEL)

TEXT.build_vocab(trainset, min_freq = 5)
LABEL.build_vocab(trainset)

class BasicGRU(nn.Module):
  def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
    super(BasicGRU, self).__init__()
    self.n_layers = n_layers
    self.embed = nn.Embedding(n_vocab, embed_dim) # n_vocab 개수의 단어들을 embed_dim 크기의 벡터로 Embedding
    self.hidden_dim = hidden_dim
    self.dropout = nn.Dropout(dropout_p)
    self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
    self.out = nn.Linear(self.hidden_dim, n_classes)

  def forward(self, x):
    x = self.embed(x)
    x , _ = self.gru(x)
    h_t = x[:, -1, :]

    self.dropout(h_t)
    out = self.out(h_t)

    return out

  # Hyperparameters를 변경해가며 학습할 것


import random

Valid_Max = 0
for i in range(36):  # Total 36개의 조합

  batch_array = [16, 32, 64, 128, 256, 512]
  learning_array = [0.001, 0.01, 0.1, 0.005, 0.05, 0.5]
  batch_size = random.sample(batch_array, 1)[0]
  learning_rate = random.sample(learning_array, 1)[0]

  print("")
  print("")
  print(
    f"# ----------------------------- {i} 번 째 조합 :  Batch Size : {batch_size} ||  learing_rate : {learning_rate} ----------------------------------------- #")

  train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (trainset, validset, testset), batch_size=batch_size, shuffle=True, repeat=False
  )

  vocab_size = len(TEXT.vocab)
  n_classes = 3  # Positive / Negative

  model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(device)
  criterion = torch.nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  Flag = True
  Valid_current = 0
  Valid_count = 0
  epoch = 0
  while (Flag == True):
    epoch += 1
    avg_cost = 0
    for batch in train_iter:
      X, Y = batch.text.to(device), batch.label.to(device)
      Y.data.sub_(1)

      hypothesis = model(X)
      cost = criterion(hypothesis, Y)

      optimizer.zero_grad()
      cost.backward()
      optimizer.step()

      avg_cost += cost / batch_size

    print(f"Epoch 종료 ... Valid ACC 계산 시작 | Cost : {avg_cost}")

    # 1 Epoch 끝났으면 Valid Set으로 학습 수행
    corrects = 0
    for batch in val_iter:
      X, Y = batch.text.to(device), batch.label.to(device)
      Y.data.sub_(1)

      hypothesis = model(X)
      corrects += (hypothesis.max(1)[1].view(Y.size()).data == Y.data).sum()

      # ACC 출력
    print('Valid Accuracy = ', (corrects / len(val_iter.dataset) * 100.0).item())

    # 현재 Valid - 이전 Valid ACC 비교해서 값이 2번 이상 증가하지 않으면 Flag Off...
    Valid_current = (corrects / len(val_iter.dataset) * 100.0).item()
    if abs(Valid_current - Valid_Max) == 0 or Valid_current < Valid_Max:  # 2회 이상 Valid 값이 작아지거나, 학습이 진행되지 않으면 ...
      Valid_count += 1
      if Valid_count == 2:
        print(
          f"###################################### {Valid_count}회 동안 ACC 증가가 없어 {epoch} Epoch 학습 후 종료합니다. #################################")
        Flag = False  # 학습 종료
    else:
      Valid_count = 0  # 아니라면 학습 종료 Flag 초기화
    print("Valid Count : ", Valid_count)
    # 최적의 Valid Acc 저장
    if Valid_current > Valid_Max:
      Valid_Max = Valid_current
      # Best Acc가 등장하면 그 때마다 Model Save
      print(
        "# ------------------------------------------------------------ Get Best Acc ... Save Model ------------------------------------------- #")
      best_batch = batch_size
      best_learning_rate = learning_rate
      torch.save(model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/인공지능 응용/model_s1.pt')

    # Best Valid ACC
print(f"Best Valid : {Valid_Max} , Best Hyperparameter : {best_batch} & {best_learning_rate}")

# model load
model_new = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(device)
model_new.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/인공지능 응용/model_s1.pt'))

# Test
corrects = 0
for batch in test_iter:
  x,y = batch.text.to(device), batch.label.to(device)
  y.data.sub_(1)
  hypothesis = model_new(x)
  corrects += (hypothesis.max(1)[1].view(y.size()).data == y.data).sum()

print('Test Accuracy = ', (corrects/len(test_iter.dataset)*100.0).item())