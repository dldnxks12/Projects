import numpy as np
import gym
import random
from time import sleep
#import matplotlib.pyplot as plt


env = gym.make('BipedalWalker-v3')
env.reset()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))


done = False
for i in range(300):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

env.close()