'''

# 1)
plt.ion()  # 그때 그때 그림을 갱신하고 싶다.

# 2)
plt.ioff() # 그림에 관련된 모든 명령을 아래에서 다 실행한 후
 ~~
plt.show() # 최종적으로 그림을 한 번만 그린다.

'''
import gym
import sys
import math
import random
import numpy as np
import collections

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

from time import sleep
from itertools import count
from collections import deque
from IPython.display import clear_output
from PIL import Image

#########################################################################
# 기본 환경 Setting ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("")
print(f"# -- On {device} -- #")
print("")

#########################################################################
# Model Setting ...
class ReplayBuffer():
  def __init__(self):
    self.buffer = collections.deque(maxlen = 50000)

  def put(self, transition):
    self.buffer.append(transition)

  def sample(self, n):
    mini_batch = random.sample(self.buffer, n)
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for transition in mini_batch:
      state, action, reward, next_state, done = transition
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      next_states.append(next_state)
      dones.append(done)

    return states, actions, rewards, next_states, dones

  def size(self):
    return len(self.buffer)

class DQN(nn.Module):
  def __init__(self, w, h, outputs):
    super().__init__()
    # In Channel = 3 | Out Channel = 16
    self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size = 5, stride = 2)
    self.bn3 = nn.BatchNorm2d(32)

    # Calculate Convolution Output Size
    def conv2d_size_out(size, kernel_size = 5, stride = 2):
      return (size - (kernel_size - 1) - 1) // stride + 1 # Convolution Output Size

    # Convolution을 3번 거친 결과의 Width & Height - fc layer input 크기 계산하는 쉬운 방법 get
    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
    linear_input_size = convw*convh*32
    self.head = nn.Linear(linear_input_size , outputs)

  def forward(self, x):
    x = x.to(device)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = self.head(x.view(x.size(0), -1)) # data flatten 후 fc layer에 넣어주기

    return x

#########################################################################
# Image Preprocessing ...
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# Image Preprocessing ...
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

# Image Preprocessing ...
def get_screen():
    # Rendering Image 가져오기
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Channel 제외 height, width Get
    _, screen_height, screen_width = screen.shape
    # Channel 제외 Height, Width 조정
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]

    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return resize(screen).unsqueeze(0) # 개수 x 채널 x 높이 x 넓이

#########################################################################
# Env setting ...
env = gym.make('CartPole-v1').unwrapped # Env의 모든 성질을 가져다 수정해서 쓰려면 unwrapped
env.reset()

BATCH_SIZE = 128 # 한 번에 학습시킬 Image 개수
GAMMA = 0.999    # Discount Factor
EPS = 0.1        # Soft greedy ...

# Get Cropped Image
init_screen = get_screen()

# Image shape
# init_screen.shape : [1, 3, 40, 90]
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

# Q Network의 출력 : Q(. , 1) , Q (. , 2) 총 2개
Q        = DQN(screen_height, screen_width, n_actions).to(device)
Q_target = DQN(screen_height, screen_width, n_actions).to(device)

# Parameter 동기화
Q_target.load_state_dict(Q.state_dict())
Q_target.eval()

optimizer = optim.Adam(Q.parameters(), lr = 0.001)
memory = ReplayBuffer()

# Update method 1
def update(net, net_target):
  for param_target, param in zip(net_target.parameters(), net.parameters()):
    param_target.data.copy_(param.data)

# Update method 2
def update2(net, target_net):
    target_net.load_state_dict(net.state_dict())

def select_action(state):
    sample = random.random()
    if sample > EPS: # sample > 0.1
        with torch.no_grad(): # 그냥 뽑기만 할 거니까 학습 xxx
            # Max Q value의 Index 1겹 씌워서 return
            return Q(state).max(1)[1].view(1, 1)
    else:
        # action range에서 무작위로 선택
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

########################################################################
# Optimize

def Optimize():
    if memory.size() < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # print(np.shape(states)) # 128 x  ...
    # print(np.shape(state))  # 1 x 3 x 40 x 90

    # 128개의 현재 State Sample에서 None이 아닌 것들만 골라내기  ...
    non_final_mask = torch.tensor(tuple(map(lambda x : x is not None, next_states)), device = device, dtype = torch.bool)

    # 128개의 다음 State Sample에서 None이 아닌 것들 골라서 버리기
    non_final_next_states = torch.cat([x for x in next_states if x is not None])

    batch_states = torch.cat(states)   # 128 x 3 x 40 x 90
    batch_actions = torch.cat(actions) # 128 x 1 (Index .. )
    batch_rewards = torch.cat(rewards) # 128

    # Batch_actions의 Index에 해당하는 Q Value들 ...
    Q_Values = Q(batch_states).gather(1, batch_actions)

    # None Mask에 의해서, 학습되지 않는 놈들은 0의 값으로 남아있을 것 ..
    next_state_values = torch.zeros(BATCH_SIZE , device = device)
    next_state_values[non_final_mask] = Q_target(non_final_next_states).max(1)[0].detach()

    Q_target_Values = batch_rewards + (next_state_values * GAMMA)

    criterion = nn.SmoothL1Loss()
    loss = criterion(Q_Values, Q_target_Values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
#    Gradient Clamping --- Loss Nan 방지
#    for param in Q.parameters():
#        param.grad.data.clamp_(-1, 1)
    optimizer.step()


########################################################################
# Training ...

MAX_EPISODE = 5000
episode = 0
scores = []
while episode < MAX_EPISODE:
    score = 0
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()

    # 이전 Image와 현재 Image의 Pixel간 차이를 State로 ...
    state = current_screen - last_screen

    for t in count():
        action = select_action(state)
        # next_state, reward, done, info = env.step() ...
        _, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], device = device) # State, Action들과 같은 형태로 만들고 GPU에 올리기

        last_screen = current_screen
        current_screen = get_screen()

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None # Episode가 끝나면 현재와 이전 Pixel간 차이 None으로 ...

        memory.put((state, action, reward, next_state, done))
        state = next_state
        score += reward

        # Optimize ..
        Optimize()

        # if Done .. Out
        if done:
            break

    if episode % 10 == 0:
        update2(Q, Q_target)

    episode += 1
    scores.append(score)
    print(f"Episode : {episode} || Score : {score}")

env.close()
print("Story End..")



