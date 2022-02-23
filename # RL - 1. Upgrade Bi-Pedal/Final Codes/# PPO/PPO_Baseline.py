'''
L_clip loss function을 최적화할 것
'''
import gym
import sys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from IPython.display import clear_output
from time import sleep

from collections import deque

# Hyperparameters
lr    = 0.0005  # Learning rate
gamma = 0.98    # Discount factor
lmbda = 0.95    # GAE (Generalized Advantage funtion Estimation)에 사용
eps_clip = 0.1  # L_Clip할 때 사용할 범위
K_epoch = 3     # 모아둔 데이터를 몇 번 반복해서 학습할 지
T_horizon = 200 # 몇 타임 스텝 동안 데이터를 모을 지


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(4, 256)         # DQN state : (위치, 속도, 각도, 각속도)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    # Actor
    def pi(self, x, softmax_dim=0):          # 배치 처리를 위해 softmax dimension을 지정해주어야한다.
        x = F.relu(self.fc1(x))              # Train할 때는 dimension = 1, data 모을 때는 dimenstion = 0
        x = self.fc_pi(x)

        prob = F.softmax(x, dim=softmax_dim) # softmax 확률 분포 Return --- 다음 Action에 대한 Policy
        return prob

    # Critic
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def make_batch():
    s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

    # T time step 씩 돌며 넣어놨던 덩어리들 하나씩 빼오기
    for transition in data:
        s, a, r, s_prime, prob_a, done = transition

        s_lst.append(s)
        a_lst.append([a])
        r_lst.append([r])
        s_prime_lst.append(s_prime)
        prob_a_lst.append([prob_a])
        done_mask = 0 if done else 1
        done_lst.append([done_mask])

    s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float)     ,      \
                                          torch.tensor(a_lst)                        ,      \
                                          torch.tensor(r_lst, dtype=torch.float)     ,      \
                                          torch.tensor(s_prime_lst,dtype=torch.float),      \
                                          torch.tensor(done_lst, dtype=torch.float)  ,      \
                                          torch.tensor(prob_a_lst)

    return s, a, r, s_prime, done_mask, prob_a


def train(model, optimizer):
    states, actions, rewards, next_states, done_masks, action_probabilities = make_batch()

    # 같은 데이터에 대해 K번 학습을 진행
    for i in range(K_epoch):
        td_target = rewards + gamma * model.v(next_states) * done_masks # 배치 처리
        delta = td_target - model.v(states)
        delta = delta.detach().numpy() # Detach : Gradient Graph를 떼어내겠다

        # 위에서 얻은 delta array를 이용해서 GAE 구하기
        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]: # 거꾸로 하나씩 가져와서 Recursive하게 GAE 계산
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)

        pi = model.pi(states, softmax_dim=1) # for loop에서 Model을 Update 하면서 Policy 뽑아내기
        pi_a = pi.gather(1, actions)         # 해당하는 Action에 대한 New Policy의 확률들

        # Loss function에 사용할 Ratio 구하기
        # new_policy/old_policy == exp(log(new_policy)-log(old_policy)) --- 이렇게 바꾸면 계산이 더 효율적임
        # action_probabilities : 경험을 쌓을 때 사용했던 Policy (Old)
        # pi_a                 : 현재 최신 Policy (New)
        ratio = torch.exp(torch.log(pi_a) - torch.log(action_probabilities))

        # 2 개의 Surrogate function
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

        # Loss Function = Policy Loss(maximize) + Value Loss(minimize)
        loss1 = -torch.min(surr1, surr2)
        loss2 = F.smooth_l1_loss(model.v(states), td_target.detach()) # detach()가 매우매우 중요

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()


# env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v0')
ppo = PPO()
optimizer = optim.Adam(ppo.parameters(), lr=lr)

MAX_EPISODES = 3000
score = 0.0
print_interval = 20
reward_history = []
reward_history_100 = deque(maxlen=100)

for episode in range(MAX_EPISODES):
    s = env.reset()
    done = False
    data = []
    while not done:
        # T Step만큼 데이터를 모으고 학습  (n Step TD와 비슷)
        for t in range(T_horizon):

            # Policy Return
            prob = ppo.pi(torch.from_numpy(s).float())
            # Policy에 따라 Action 선택
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)

            # memory에 넣는게 아니라 data 통에 넣기 , 기존과 다른 것은 그 Action에 대한 확률도 넣어준다.
            # 나중에 Ratio를 계산할 때 사용할 것이기 때문 ! Ratio : (PI_new / PI_old)
            data.append((s, a, r / 100.0, s_prime, prob[a].item(), done))
            s = s_prime

            score = score + r
            if done:
                break

        train(ppo, optimizer)
        data = [] # data 통 비우기

    reward_history.append(score)
    reward_history_100.append(score)
    avg = sum(reward_history_100) / len(reward_history_100)
    episode = episode + 1
    if episode % 100 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, score, avg))

    score = 0.0

env.close()
