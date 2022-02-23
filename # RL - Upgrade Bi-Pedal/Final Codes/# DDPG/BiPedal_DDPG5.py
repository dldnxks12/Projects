# 코드 동작 OK
# 학습 X
###########################################################################
# To Avoid Library Collision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
###########################################################################

import gym
import sys
import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from time import sleep
from collections import deque
import matplotlib.pyplot as plt

###########################################################################
# Env Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"On {device}")
print("")

# Hyperparameters
lr_mu = 0.05          # Learning Rate for Torque (Action)
lr_q  = 0.1          # Learning Rate for Q
gamma = 0.99          # discount factor
batch_size = 64       # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 20000  # Replay Memory Size
tau = 0.05            # for target network soft update

###########################################################################
# Model and ReplayBuffer
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition) # transition : (state, action, reward, next_state, done)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) # buffer에서 n개 뽑기
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s) # s = [COS SIN 각속도]
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)

        return torch.tensor(s_lst, device = device , dtype=torch.float), torch.tensor(a_lst, device = device , dtype=torch.float), \
               torch.tensor(r_lst, device = device , dtype=torch.float), torch.tensor(s_prime_lst, device = device , dtype=torch.float), \
               torch.tensor(done_mask_lst, device = device , dtype=torch.float)

    def size(self):
        return len(self.buffer)

# Deterministic Policy ...
'''
    ### Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4 joints at both hips and knees.
'''
class MuNet(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(24, 64) # Input  : 24 continuous states
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, 4) # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s   = nn.Linear(24, 64)    # State  24 개
        self.fc_a   = nn.Linear(4, 64)     # Action 4  개
        self.fc_q   = nn.Linear(128, 64)  # State , Action 이어붙이기
        self.fc_out = nn.Linear(64, 32)   # Output : Q value
        self.fc_out2 = nn.Linear(32, 1)    # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x)) # 128
        h2 = F.relu(self.fc_a(a)) # 128
        cat = torch.cat([h1, h2], dim = 1)  # 256
        q = F.relu(self.fc_q(cat))   # 128
        q = self.fc_out(q)   # 64
        q = self.fc_out2(q)  # 1 - Q Value
        return q


# Action에 Noise를 추가하여 Exploration ---- 이후 앙상블로 바꿔볼 것
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

###########################################################################
# Train ...
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    Critic, Actor = 0.0 , 0.0

    y = rewards + ( gamma * q_target(next_states, mu_target(next_states)) * dones )
    Critic = torch.nn.functional.smooth_l1_loss( q(states, actions), y.detach() )
    q_optimizer.zero_grad()
    Critic.backward()
    q_optimizer.step()

    Actor = -q(states, mu(states)).mean()
    mu_optimizer.zero_grad()
    Actor.backward()
    mu_optimizer.step()


# Soft Update
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


# state  : continuous 24 state
# action : continuous 4 action
env = gym.make('BipedalWalker-v3')
env.reset()

memory = ReplayBuffer()

# 2개의 동일한 네트워크 생성 ...
q =  QNet().to(device)
q_target = QNet().to(device)
mu = MuNet().to(device)
mu_target = MuNet().to(device)

q_target.load_state_dict(q.state_dict())   # 파라미터 동기화
mu_target.load_state_dict(mu.state_dict()) # 파라미터 동기화

q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(4))
MAX_EPISODES = 3000

reward_history_20 = []

'''
    ### Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4 joints at both hips and knees.
'''

E = 0.001
episode = 0
last_action  = 0
while episode < MAX_EPISODES:
    state = env.reset()
    done = False
    score = 0.0
    lasting_time = 1
    while not done:
        action = mu(torch.from_numpy(state).to(device))
        noise = torch.tensor(ou_noise(), device = device)

        # Add Exploration property
        action = (action + noise).cpu().detach().numpy()
        next_state, reward, done, _ = env.step(action)

        # Modify Reward
        if abs(action - last_action).mean() < E: # Action의 변화가 적다면
            done = True # Episode 바로 중단
            lasting_time = 1
        else:
            lasting_time += 0.0005
            if reward <= 0:
                reward = reward
            else:
                reward = reward * 10 * lasting_time

        # Type Check
        # print(type(state), type(action), type(next_state), type(reward), type(done))
        memory.put((state, action, reward, next_state, done))
        score += reward
        state = next_state
        last_action = action

    if memory.size() > 1000:
        for _ in range(10):
            train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
            soft_update(mu, mu_target)
            soft_update(q, q_target)

    if episode % 20 == 0:
        reward_history_20.append(score)
        # avg = sum(reward_history_20) / len(reward_history_20)
        print('episode: {}, reward: {:.1f}'.format(episode, score))
    episode += 1

env.close()

#######################################################################
# Record Hyperparamters & Result Graph

with open('DDPG_5.txt', 'w', encoding = 'UTF-8') as f:
    f.write("# ----------------------- # " + '\n')
    f.write("DDPG_Parameter 2022-2-12" + '\n')
    f.write('\n')
    f.write('\n')
    f.write("# - Extra ViewPoints - #" + '\n')
    f.write('\n')
    f.write("Fast Reset Version" + '\n')
    f.write("Train x 10 Version" + '\n')
    f.write('\n')
    f.write('\n')
    f.write("# - Category 1 - #" + '\n')
    f.write('\n')
    f.write("Reward        : Sign Assigned" + '\n')
    f.write("lr_mu         : " + str(lr_mu) + '\n')
    f.write("lr_q          : " + str(lr_q) + '\n')
    f.write("tau           : " + str(tau) + '\n')
    f.write('\n')
    f.write("# - Category 2 - #" + '\n')
    f.write('\n')
    f.write("batch_size    : " + str(batch_size)   + '\n')
    f.write("buffer_limit  : " + str(buffer_limit) + '\n')
    f.write("memory.size() : 2000" + '\n')
    f.write("# ----------------------- # " + '\n')

length = np.arange(len(reward_history_20))
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG")
plt.plot(length, reward_history_20)
plt.savefig('DDPG_5.png')