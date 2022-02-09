# 학습 OK
# D3QN : Double Deep Q Learning + Dueling Architecture
# Double Q와 Double Deep 각각 보다 학습 속도 빠름

import gym
import sys
import math
import random
import random
import numpy as np
import torch
import collections
from collections import deque
from time import sleep
from IPython.display import clear_output

##########################################################################
# Env Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("")
print(f"On {device}")
print("")

##########################################################################
# Buffer 생성
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            done_mask = 0.0 if done else 1.0
            dones.append([done_mask])

        return torch.tensor(states, device=device, dtype=torch.float), torch.tensor(actions, device=device,dtype=torch.float), torch.tensor(rewards, device=device, dtype=torch.float), torch.tensor(next_states, device=device,dtype=torch.float), torch.tensor(dones,device=device,dtype=torch.float)

    def size(self):
        return len(self.buffer)

##########################################################################
# Model 생성
class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcA1 = torch.nn.Linear(4, 32)
        self.fcA2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = self.fcA1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcA2(x)
        x = torch.nn.functional.softmax(x, dim=-1)
        return x  # Return Policy


class HNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcQ1 = torch.nn.Linear(4,  128)
        self.fcQ2 = torch.nn.Linear(128, 64)
        self.fcQ3 = torch.nn.Linear(64,  32)

        self.fc_tail = torch.nn.Linear(32, 1)  # H(s)

    def forward(self, x):
        x = self.fcQ1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ3(x)
        x = torch.nn.functional.relu(x)

        x = self.fc_tail(x)  # H(s)
        return x

class SNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcQ1 = torch.nn.Linear(4,  128)
        self.fcQ2 = torch.nn.Linear(128, 64)
        self.fcQ3 = torch.nn.Linear(64,  32)

        self.fc_tail = torch.nn.Linear(32, 2)  # S(s,a)

    def forward(self, x):
        x = self.fcQ1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ3(x)
        x = torch.nn.functional.relu(x)

        x = self.fc_tail(x)  # S(s, a)
        return x


##########################################################################
# Model & Hyperparameters
memory = ReplayBuffer()
alpha = 0.001
gamma = 0.99

MAX_EPISODE = 10000
episode = 0

# Policy Update
pi = PolicyNetwork().to(device)

# H , S Updatae
H = HNetwork().to(device)
S = SNetwork().to(device)
H_target = HNetwork().to(device)
S_target = SNetwork().to(device)
H_target.load_state_dict(H.state_dict())  # Synchronize Parameters
S_target.load_state_dict(S.state_dict())  # Synchronize Parameters

pi_optimizer = torch.optim.Adam(pi.parameters(), lr=alpha)

fc1 = list(H.parameters())
fc2 = list(S.parameters())
optimizer = torch.optim.Adam(fc1 + fc2, lr=alpha)

##########################################################################

# Soft Update
def update(H, S , H_target, S_target):
    for param_target, param in zip(H_target.parameters(), H.parameters()):
        param_target.data.copy_(param.data)
    for param_target, param in zip(S_target.parameters(), S.parameters()):
        param_target.data.copy_(param.data)

# Train ...
def train(memory, H, S , H_target, S_target, optimizer):
    states, actions, rewards, next_states, dones = memory.sample(16)

    loss = 0
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        action = int(action.item())
        if done == 0:
            y = reward
        else:
            with torch.no_grad():
                Q_target = H_target(next_state) + S_target(next_state) - (S_target(next_state).mean())
                result = H(state) + S(state) - (S(state).mean())
                A = torch.argmax(H(state) + S(state) - (S(state).mean())).item()
            y = reward + gamma * (Q_target[A])

        Q = H(state) + S(state)[action] - (S(state).mean())
        loss += (y - Q) ** 2

    loss = loss / 16

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env = gym.make('CartPole-v1')
while episode < MAX_EPISODE:

    state = env.reset()
    done = False
    score = 0

    while not done:
        state = np.array(state)
        policy = pi(torch.from_numpy(state).float().to(device))

        if np.random.rand() <= 0.1:
            action = random.randrange(env.action_space.n)
        else:
            with torch.no_grad():
                Q = H(torch.from_numpy(state).to(device)) + S(torch.from_numpy(state).to(device)) - S(torch.from_numpy(state).to(device)).mean()
            action = torch.argmax(Q).item()

        next_state, reward, done, info = env.step(action)

        memory.put((state, action, reward, next_state, done))  # Stack ...

        score += reward
        state = next_state

        if memory.size() > 2000:
            train(memory, H, S , H_target, S_target, optimizer)

            if episode % 10 == 0:
                update(H, S , H_target, S_target)

    print(f"Epidoe : {episode} || Reward : {score}")
    episode += 1

env.close()
