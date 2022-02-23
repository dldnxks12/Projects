# 학습 OK
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
import torch.nn.functional as F
import torch.optim as optim
from time import sleep
from collections import deque
import matplotlib.pyplot as plt

#GPU Setting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("")
print(f"On {device}")
print("")

# Hyperparameters
lr_mu = 0.0005       # Learning Rate for Torque (Action)
lr_q = 0.001         # Learning Rate for Q
gamma = 0.99         # discount factor
batch_size = 16      # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 50000 # Replay Memory Size
tau = 0.005          # for target network soft update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition) # transition : (state, action, reward, next_state, done)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) # buffer에서 n개 뽑기
        states, actions, rewards, next_states, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            states.append(state) # s = [COS SIN 각속도]
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_lst = np.array(states)
        #a_lst = np.array(actions)
        r_lst = np.array(rewards)
        s_prime_lst = np.array(next_states)
        done_mask_lst = np.array(done_mask_lst)

        return torch.tensor(s_lst, device = device, dtype=torch.float), torch.tensor(actions, device = device,dtype=torch.float), \
               torch.tensor(r_lst, device = device,dtype=torch.float), torch.tensor(s_prime_lst,device = device, dtype=torch.float), \
               torch.tensor(done_mask_lst, device = device, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):  # Mu = Torque -> Action
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x): # Input : state (COS, SIN, 각속도)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2
        return mu # Return Deterministic Policy

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)   # State = (COS, SIN, 각속도)
        self.fc_a = nn.Linear(1, 64)   # Action = Torque
        self.fc_q = nn.Linear(128, 32) # State , Action 이어붙이기
        self.fc_out = nn.Linear(32, 1) # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x)) # 64
        h2 = F.relu(self.fc_a(a)) # 64
        cat = torch.cat([h1, h2], dim = 1)  # 128
        q = F.relu(self.fc_q(cat)) # 32
        q = self.fc_out(q)  # 1
        return q


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


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    Q_loss, mu_loss = 0, 0

    y = rewards + ( gamma*q_target(next_states, mu_target(next_states)) ) * dones

    Q_loss = torch.nn.functional.smooth_l1_loss(q(states, actions) , y.detach())
    q_optimizer.zero_grad()
    Q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(states, mu(states)).mean()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

env = gym.make('Pendulum-v1')
memory = ReplayBuffer()

# 2개의 동일한 네트워크 생성 ...
q =  QNet().to(device)
q_target = QNet().to(device)
mu = MuNet().to(device)
mu_target = MuNet().to(device)

q_target.load_state_dict(q.state_dict())   # 파라미터 동기화
mu_target.load_state_dict(mu.state_dict()) # 파라미터 동기화

score = 0.0
avg_history = []
reward_history_20 = deque(maxlen=100)

mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
MAX_EPISODES = 500

# Action Space Map
A = np.arange(-2, 2, 0.001)
for episode in range(MAX_EPISODES):
    s = env.reset()
    done = False
    score = 0.0

    while not done: # Stacking Experiences

        a = mu(torch.from_numpy(s).float().to(device)) # Return action (-2 ~ 2 사이의 torque  ... )

        # Discretize Action Space (A = np.arange(-2, 2, 0.001))
        discrete_action = np.digitize(a.cpu().detach().numpy(), bins = A)

        # Soft Greedy Policy
        sample = random.random()
        if sample < 0.1:
            random_action = np.array([random.randrange(0, len(A))])
            action = A[random_action - 1]
        else:
            action = A[discrete_action - 1]

        s_prime, r, done, info = env.step(action)
        memory.put((s, a, r / 100.0, s_prime, done))
        score = score + r
        s = s_prime

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

    # Moving Average Count
    reward_history_20.insert(0, score)
    if len(reward_history_20) == 10:
        reward_history_20.pop()
    avg = sum(reward_history_20) / len(reward_history_20)
    avg_history.append(avg)
    if episode % 10 == 0:
        print('episode: {} | reward: {:.1f} | 10 avg: {:.1f} '.format(episode, score, avg))
    episode += 1

env.close()

# Record Hyperparamters & Result Graph
with open('DDPG_Discretization.txt', 'w', encoding = 'UTF-8') as f:
    f.write("# ----------------------- # " + '\n')
    f.write("Parameter 2022-2-12" + '\n')
    f.write('\n')
    f.write('\n')
    f.write("# - Category 1 - #" + '\n')
    f.write('\n')
    f.write("Reward        : Basic Env Setting" + '\n')
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

length = np.arange(len(avg_history))
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG_Discretization")
plt.plot(length, avg_history)
plt.savefig('DDPG_Discretization.png')


avg_history = np.array(avg_history)
np.save("./ddpg_dis_save1", avg_history)