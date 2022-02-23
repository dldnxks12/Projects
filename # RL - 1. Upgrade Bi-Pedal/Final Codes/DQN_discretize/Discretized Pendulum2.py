###########################################################################
# To Avoid Library Collision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
    def __init__(self, height, width, output_action):
        super().__init__()
        # Channel : In = 3, Out = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size= 3, stride = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size= 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(32)

        # Calculate Convolution Output Size
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
          return (size - (kernel_size - 1) - 1) // stride + 1 # Convolution Output Size

        # Convolution을 3번 거친 결과의 Width & Height - fc layer input 크기 계산하는 쉬운 방법 get
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = convw*convh*32
        self.head = nn.Linear(linear_input_size , output_action)

    def forward(self , x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = self.head(x)
        x = torch.tanh(x) * 2
        return x

# 학습하기 편하게 Pendulum 위치만 자르기
def get_screen():
    # 500 x 500 x 3
    screen = env.render(mode = 'rgb_array')

    # 260 x 260 x 3
    Window = screen[120:380, 120:380, :]

    # 3 x 130 x 130
    # Window = cv2.resize(Window, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA).transpose((2,0,1))

    # 3 x 26 x 26
    Window = cv2.resize(Window, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA).transpose((2, 0, 1))

    # 1 x 3 x 26 x 26
    Window = torch.tensor(Window, dtype = torch.float).unsqueeze(0)

    # Window
    # Type  : torch
    # Shape : 1 x 3 x 26 x 26
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
A = np.arange(-2, 2, 0.00001)
steps_done = 0
def action_selection(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    a = Q(state.to(device))
    discrete_action = np.digitize(a.cpu().detach().numpy(), bins = A)


    if sample < eps_threshold:
        action = A[discrete_action - 1]
    else:
        # Random 선택
        random_action = np.array([random.randrange(0, len(A))])
        action = np.expand_dims(A[random_action - 1], axis = 0)


    action = torch.from_numpy(action)
    # Action
    # Type  : pytorch
    # Shape : [1,1]

    return action
##########################################################################
# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32 # 한 번에 학습시킬 Image 개수
GAMMA = 0.9    # Discount Factor
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Get Cropped Image
init_screen = get_screen()
_, _, window_height, window_width = init_screen.shape

# Window 넣어주고 1개 Action 받아올 것
# Output -> Discrete Action으로 Mapping 시킬 것
Q        = DQN(window_height, window_width, 1).to(device)
Q_target = DQN(window_height, window_width, 1).to(device)

Q_target.load_state_dict(Q.state_dict())
Q_target.eval() # 내가 원할 때 학습 시킬 것

optimizer = optim.Adam(Q.parameters(), lr = LEARNING_RATE)
memory = ReplayBuffer()


def Optimize():
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # states      # 32 x ...
    # next_states # 32 x ...
    # state       # 1 x 3 x 130 x 130

    # actions     # 32 x ...
    # action      # 1 x 1


    non_final_mask = torch.tensor(tuple(map(lambda x : x is not None, next_states)), device= device, dtype = torch.bool)
    non_final_next_states = torch.cat([x for x in next_states if x is not None]).to(device)

    batch_states = torch.cat(states).to(device)   # 32 x 3 x 130 x 130
    batch_actions = torch.cat(actions).to(device) # 32 x 1
    batch_rewards = torch.cat(rewards).to(device)

    Q_values = Q(batch_states).squeeze(1) # size : [32]
    next_state_values = torch.zeros(BATCH_SIZE, device = device)
    next_state_values[non_final_mask] = Q_target(non_final_next_states).squeeze(1).detach() # size : [32]

    Q_target_values = batch_rewards + GAMMA*next_state_values

    loss = 0
    criterion = nn.SmoothL1Loss()
    loss = criterion(Q_values, Q_target_values)

    optimizer.zero_grad()
    loss.backward()
    # Gradient Clamping --- Loss Nan 방지
    #for param in Q.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()

##########################################################################

score = 0.0
reward_history_20 = deque(maxlen=100)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
MAX_EPISODES = 500
episode = 0
while episode < MAX_EPISODES:
    env.reset()

    last_window = get_screen()
    current_window = get_screen()

    # Pixel 간 차이를 State로  ...
    state = current_window - last_window
    done = False
    score = 0.0
    simul_step = 0

    while not done: # Stacking Experiences
        action = action_selection(state)
        _ , reward, done, info = env.step(action)

        reward = torch.tensor([reward] , device = device)

        last_window = current_window
        current_window = get_screen()

        if not done:
            next_state = current_window - last_window
        else:
            # print(simul_step)
            # print(done)
            next_state = None

        # Method 1
        memory.put((state, action, reward / 10.0, next_state, done))
        state = next_state
        score += reward

        # Method 2
        # Stack을 이용해서 4 장의 frame 넣어주기 - Atari Game 참고

        if memory.size() > 500:
            for _ in range(10):
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
with open('DQN_Discretization2.txt', 'w', encoding = 'UTF-8') as f:
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
plt.title("DQN_Discretization2")
plt.plot(length, reward_history_20)
plt.savefig('DQN_Discretization2.png')
