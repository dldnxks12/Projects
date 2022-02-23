###########################################################################
# To Avoid Library Collision
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
###########################################################################

import cv2
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


##########################################################################
#GPU Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("")
print(f"On {device}")
print("")

env = gym.make('Pendulum-v1').unwrapped # Env의 모든 성질을 가져다 수정해서 쓰려면 unwrapped
env.reset()

##########################################################################
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen = 50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        states, actions, action_indexes, rewards, next_states, dones = [], [], [], [], [], []

        for transition in mini_batch:
            state, action, action_index, reward, next_state, done = transition
            states.append(state)
            actions.append(action)
            action_indexes.append(action_index)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, action_indexes, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

class Block(nn.Module):

    def __init__(self, in_channels = 10):
        super().__init__()
        # Block 1
        self.branch1_1 = nn.Conv2d(in_channels, 16, kernel_size = 1)

        # Block 2
        self.branch3_1 = nn.Conv2d(in_channels , 16, kernel_size = 1)
        self.branch3_2 = nn.Conv2d(16, 24, kernel_size = 3, padding = 1)
        self.branch3_3 = nn.Conv2d(24, 24, kernel_size = 3, padding = 1)

        # Block 3
        self.branch5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)  # 1x1 Conv
        self.branch5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # Block 4
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        # Block 1
        branch1x1 = self.branch1_1(x) # torch.Size([1, 16, 26, 26])

        # Block 2
        branch3x3 = self.branch3_1(x)
        branch3x3 = self.branch3_2(branch3x3)
        branch3x3 = self.branch3_3(branch3x3) # torch.Size([1, 24, 26, 26])

        # Block 3
        branch5x5 = self.branch5_1(x)
        branch5x5 = self.branch5_2(branch5x5) # torch.Size([1, 24, 26, 26])

        # Block 4
        branch_pool = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
        branch_pool = self.branch_pool(branch_pool) # torch.Size([1, 24, 26, 26])
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]

        # 1 x 88 x 26 x 26
        x = torch.cat(outputs, 1)
        return x

######################################################################## frame 추가하기
class Inception(nn.Module):
    def __init__(self, output_action):
        super(Inception, self).__init__()

        # In Channel : Number of Stacked Frames ...
        self.conv1 = nn.Conv2d(2, 10, kernel_size =5) # 1 x 88 x 26 x 26
        self.bn1   = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.bn2   = nn.BatchNorm2d(20)

        self.incept1 = Block(in_channels=10)
        self.incept2 = Block(in_channels=20)

        self.mp = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(352, output_action)

    def forward(self, x):

        in_size = x.size(0)  # 0 차원 크기 : batch size
        x = self.bn1(self.conv1(x))           # torch.Size([1, 10, 22, 22])
        x = F.relu(self.mp(x))      # torch.Size([1, 10, 11, 11])
        x = self.incept1(x)         # torch.Size([1, 88, 11, 11])

        x = self.bn2(self.conv2(x))           # torch.Size([1, 20, 7, 7])
        x = F.relu(self.mp(x))      # torch.Size([1, 20, 3, 3])
        x = self.incept2(x)         # torch.Size([1, 88, 3, 3])

        x = x.reshape(in_size, -1)   # torch.Size([1, 792])
        x = self.fc(x)               # torch.Size([1, 80])

        x = F.softmax(x, dim = 1)  # [1 , 80]

        return x

# 학습하기 편하게 Pendulum 위치만 자르기
def get_screen():
    # 500 x 500 x 3
    screen = env.render(mode = 'rgb_array')

    # 260 x 260 x 3
    Window = screen[150:350, 150:350, :]

    # 3 x 26 x 26
    # 3 x 20 x 20
    Window = cv2.resize(Window, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

    Window = cv2.cvtColor(Window, cv2.COLOR_BGR2GRAY)

    # 26 x 26 # Gray Color
    Window = torch.tensor(Window, dtype = torch.float)

    # Window
    # Type  : torch
    # Shape : 26 x 26
    return Window

def update(net, target_net):
    target_net.load_state_dict(net.state_dict())

# Noise 생성
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

# Action Space Map
A = np.arange(-2, 2, 0.05)
def action_selection(state):

    action_index = torch.argmax(Q(state)).unsqueeze(0)  # [ 1 , 16 ] output
    action = np.array(A[action_index - 1])
    action = np.expand_dims(action, axis = 0)
    action = np.expand_dims(action, axis = 0)

    action = torch.from_numpy(action)
    # Action
    # Type  : pytorch
    # Shape : [1,1]

    return action, action_index

##########################################################################
# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32 # 한 번에 학습시킬 Image 개수
GAMMA = 0.9    # Discount Factor

# Window 넣어주고 1개 Action 받아올 것
# Output -> Discrete Action으로 Mapping 시킬 것
Q        = Inception(len(A)).to(device)
Q_target = Inception(len(A)).to(device)

Q_target.load_state_dict(Q.state_dict())
Q_target.eval() # 내가 원할 때 학습 시킬 것

optimizer = optim.Adam(Q.parameters(), lr = LEARNING_RATE)
memory = ReplayBuffer()

def Optimize():
    states, actions, action_indexes, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    # states      # 32 x ...
    # next_states # 32 x ...
    # state       # 1 x 3 x 130 x 130

    # actions     # 32 x ...
    # action      # 1 x 1

    loss = 0
    for (state, action, action_index, reward, next_state, done) in zip(states, actions, action_indexes, rewards, next_states, dones):
        # Q(next_state) : 1 x 80
        A = torch.argmax(Q(next_state)) # Max Index return
        target = reward +(GAMMA * Q_target(next_state).squeeze(0)[A]) * done
        loss += (target - Q(state).squeeze(0)[action_index])**2

    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

##########################################################################
score = 0.0
reward_history_20 = deque(maxlen=100)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
MAX_EPISODES = 500
episode = 0
while episode < MAX_EPISODES:
    env.reset()

    observation = get_screen()
    state = np.array(observation)
    stack = [state] * 2
    state = np.array(stack)
    state = torch.from_numpy(state).float().to(device).unsqueeze(0)

    done = False
    score = 0.0
    simul_step = 0

    while not done: # Stacking Experiences
        action, action_index = action_selection(state)
        _ , reward, done, info = env.step(action)

        next_observation = np.array(get_screen())
        stack.pop(0)
        stack.append(next_observation)

        next_state = np.array(stack)
        next_state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)

        action_index = torch.tensor([action_index], device = device, dtype = torch.int64)
        reward = torch.tensor([reward] , device = device)

        score += reward

        memory.put((state, action, action_index, reward, next_state, done))
        state = next_state
        score += reward

        if memory.size() > 500:
            Optimize()
            update(Q, Q_target)

        # 200 번 한 후 Done
        if simul_step == 200:
            simul_step = 0
            done = True

        simul_step += 1

    score = score.cpu().detach().numpy()[0]
    reward_history_20.append(score)
    avg = sum(reward_history_20) / len(reward_history_20)

    if episode % 20 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, score, avg))
    episode = episode + 1

env.close()

# Record Hyperparamters & Result Graph
with open('DQN_Stacked.txt', 'w', encoding = 'UTF-8') as f:
    f.write("# ----------------------- # " + '\n')
    f.write("Parameter 2022-2-13" + '\n')
    f.write('\n')
    f.write('\n')
    f.write("# - Category 1 - #" + '\n')
    f.write('\n')
    f.write("Reward        : Reward / 10 " + '\n')
    f.write("LEARNING_RATE         : " + str(LEARNING_RATE) + '\n')
    f.write('\n')
    f.write("# - Category 2 - #" + '\n')
    f.write('\n')
    f.write("batch_size    : " + str(BATCH_SIZE)   + '\n')
    f.write("buffer_limit  : " + str(50000) + '\n')
    f.write("memory.size() : 500" + '\n')
    f.write("# ----------------------- # " + '\n')


length = np.arange(len(reward_history_20))
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN_Stacked")
plt.plot(length, reward_history_20)
plt.savefig('DQN_Stacked.png')
