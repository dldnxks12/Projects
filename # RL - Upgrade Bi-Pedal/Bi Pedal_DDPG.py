# 코드 동작 OK
# 학습 XX

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
lr_mu = 0.005         # Learning Rate for Torque (Action)
lr_q = 0.001          # Learning Rate for Q
gamma = 0.99         # discount factor
batch_size = 64      # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 50000 # Replay Memory Size
tau = 0.005          # for target network soft update

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
        self.fc1 = nn.Linear(24, 128) # Input  : 24 continuous states
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 4) # Output : 4 continuous actions

    def forward(self, x): # Input : state (COS, SIN, 각속도)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))  # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s   = nn.Linear(24, 64)    # State  24 개
        self.fc_a   = nn.Linear(4, 64)     # Action 4  개
        self.fc_q   = nn.Linear(128, 32)  # State , Action 이어붙이기
        self.fc_out = nn.Linear(32, 1)  # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x)) # 64
        h2 = F.relu(self.fc_a(a)) # 64

        # cat = torch.cat([h1, h2], dim = -1) # 128
        # cat = torch.cat([h1, h2], dim = 1) # 128
        cat = torch.cat([h1, h2], dim = 0)  # 128
        q = F.relu(self.fc_q(cat))   # 128
        q = self.fc_out(q)  # 1 - Q Value
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

def train(mu, mu_target, q, q_taget, memory, mu_optimizer, q_optimizer):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    Critic, Actor = 0.0 , 0.0
    for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
        if done == 0:
            y = reward
        else:
            with torch.no_grad():
                a = mu_target(next_state)
                y = reward + gamma*(q_target(next_state, a))

        Critic += (y - q(state, action))**2
    Critic = -Critic/batch_size
    q_optimizer.zero_grad()
    Critic.backward()
    q_optimizer.step()

    for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
        Actor += q(state, mu(state))

    Actor = -Actor/batch_size
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

mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(4))
MAX_EPISODES = 10000

print_interval = 20
reward_history = []
reward_history_100 = deque(maxlen=100)

episode = 0
while episode < MAX_EPISODES:
    state = env.reset()
    done = False
    score = 0.0
    while not done:
        if episode % 200 == 0:
            env.render()

        action = mu(torch.from_numpy(state).to(device))
        noise = torch.tensor(ou_noise(), device = device, dtype = torch.long)

        # Add Exploration property
        action = (action + noise).cpu().detach().numpy()

        next_state, reward, done, _ = env.step(action)

        # Type Check
        # print(type(state), type(action), type(next_state), type(reward), type(done))

        memory.put((state, action, reward, next_state, done))
        score += reward
        state = next_state

    if memory.size() > 2000:
        train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
        soft_update(mu, mu_target)
        soft_update(q, q_target)

    reward_history.append(score)
    reward_history_100.append(score)
    avg = sum(reward_history_100) / len(reward_history_100)
    if episode % 10 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, score, avg))
    episode += 1
env.close()