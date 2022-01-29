'''

# REINFORCE : 인공신경망이 가치 함수가 아닌 정책을 바로 근사한다.

인공신경망의 입력: State || 출력 : 행동  --- ㅠ(a|s)가 상태를 넣어줬을 때의 Action을 결정해주잖아?

출력층의 Activation functio : Softmax !

'''

import sys
import gym
import pylab
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.initializers import RandomUniform
import matplotlib.pyplot as plt

class REINFORCE(tf.keras.Model):
    
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = Dense(24, activation = 'relu')
        self.fc2 = Dense(24, activation = 'relu')
        self.fc_out = Dense(action_size, activation = 'softmax') # 출력층의 Activation function : Softmax

    def call(self, x): # x : state vector
        x = self.fc1(x)
        x = self.fc2(x)

        policy = self.fc_out(x) # 출력으로는 해당 State에서의 각 행동에 대한 확률을 return !!!
        return policy

class REINFORCEAgent:
    def __init__(self, state_size, action_size): # state 크기와 action 크기를 입력으로 받는 Agent Class

        self.render = True  # Drawing

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # # REINFORCE Hyperparameter 정의
        self.discount_factor = 0.99
        self.learning_rate = 0.01

        self.model = REINFORCE(self.action_size)
        self.optimizer = Adam(lr = self.learning_rate)

        # 값을 다루기 위해 빈 공간 할당
        self.states, self.actions, self.rewards = [], [], []

    # 인공신경망으로 학습한 Policy를 따라 행동을 결정
    def get_action(self, state):
        policy = self.model(state)[0] # ----------- [[위치, 속도, 각도, 각속도]] 이렇게 받을 거니까 [0] 해주기 --- 왜? NN에 넣어서 학습시키려면 차원 맞춰야지
        policy = np.array(policy) # 반환된 각 행동에 대한 확률들 numpy type으로

        # Policy 분포를 따라서 1개를 뽑자 (Optimal Policy가 아니니 [0.2 0.6] 이렇게 나올 것)
        return np.random.choice(self.action_size, 1, p = policy)[0]

    # 에피소드가 끝나면 아래 함수로 Return 값을 계산하고, 이를 이용해서 업데이트를 진행
    def discount_rewards(self, rewards): # Monte - Carlo method ...
        discounted_rewards = np.zeros_like(rewards) # rewards와 같은 크기의 영행렬
        running_add = 0

        for t in reversed(range(0, len(rewards))): # 뒤에서 부터 차레대로 ...
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards

    # 에피소드가 끝날 때 까지 상태, 행동, 보상 저장
    # 신경망을 업데이트 할 때 (theta = theta + a * ~) Q 함수대신 Return을 사용할 것이다. 따라서 이 값을 매번 저장해두어야한다.
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)

        act = np.zeros(self.action_size)
        act[action] = 1 # Make one hot Vector
        self.actions.append(act)

    def train_model(self):

        # 반환값을 그대로 사용해도 되지만, 이렇게 정규화하면 성능이 더 좋아진다.
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # CE Loss 계산
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)

            policies = self.model(np.array(self.states))
            actions = np.array(self.actions) # 크기 action_size, 값 = One hot vector

            action_prob = tf.reduce_sum(actions*policies, axis = 1) # 행동을 할 확률과, one-hot vector를 곱하면 해당하는 확률 값 얻을 수 있다.
            # CE = -sum(y*log(p)) --- y=1일 때, CE = -log(p_action)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy = - policies * tf.math.log(policies)

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

        # 다시 싹 비우기
        self.states, self.actions, self.rewards = [], [], []

        return np.mean(entropy)

if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(state_size, action_size)

    scores , episodes = [], []
    score_avg = 0

    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.append_sample(state, action, reward)
            score += reward

            reward = reward if not done or score == 500 else -1

            state = next_state

            if done:
                # 정책 신경망 업데이트
                entropy = agent.train_model()

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | X : {:.4f}m | | entropy: {:.3f}".format(
                      e, score_avg, info['X'], entropy))

                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("./save_graph/graph3.png")

                if score_avg > 500:
                    agent.model.save_weights("./model", save_format="tf")
                    sys.exit()