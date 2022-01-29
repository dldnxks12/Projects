import os
import sys
import gym
import random
import pylab
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import matplotlib.pyplot as plt


# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(tf.keras.Model):
    def __init__(self, action_size): # Action Size = 2
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size,
                            kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x) # q size = 2
        return q


# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size): # State Size : 4 , Action Size = 2
        self.render = True # Drawing

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size)        # 네트워크 모델 : 매번 Update할 것
        self.target_model = DQN(action_size) # 타켓 모델    : 가끔 한 번씩 Update할 것
        self.optimizer = Adam(lr=self.learning_rate)

        # 타깃 모델 초기화
        self.update_target_model() # 24 x 24 x 2 개 Weight

    # 네트워크 모델의 가중치를 가져와서 타깃 모델을 업데이트 ---- 가끔 업데이트 할 모델 W-
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택 --- 네트워크 모델에서 선택
    def get_action(self, state): # (1,4) [ [위치, 속도, 각도, 각속도] ]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)  # --------------------------------- 현재 State에서 q_value?
            return np.argmax(q_value[0]) # 최적의 q_value

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 미니 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        # mini_batch size ... = memory 에는 [(1x5),(1x5), ... ] 있고
        # 여기서 batch_size 만큼 가져왔다 --- [(1x5),(1x5), ... ] --- 이렇게 있겠지
        # 따라서 for sample in mini_batch --- (state, action, reward, next_state, done)의 형태

        states = np.array( [sample[0][0]     for sample in mini_batch] )
        actions = np.array([sample[1]        for sample in mini_batch])
        rewards = np.array([sample[2]        for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4]          for sample in mini_batch]) # 모델이 쓰러졌거나, 어느 선을 넘었거나 ..

        # 학습할 파라미터 = model_params ...
        '''
        with tf.GradientTape() as tape:
        
            아래 내용에서 실행되는 모든 연산을 테이프에 기록... 이 후 자동으로 역전파를 수행해주며 기록해둔 연산의 Gradient를 계산해준다.
        '''
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.model(states) # [0.2, 0.6] ~
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 상태에 대한 타깃 모델의 큐함수
            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts) # Gradient 를 계산하지 않는다.

            # 벨만 최적 방정식을 이용한 업데이트 타깃
            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts)) # tf.square(targets - predicts)가 가르키는 배열 전체 원소의 합 / 원소의 개수

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params) # d loss_d model_params 계산  Ex) [dloss_dw, dloss_db] = tape.gradient(loss, [w, b])

        # Tensorflow 에서 계산해준 값 위에 우리가 조작한 값을 덮어쓰는 방법
        self.optimizer.apply_gradients(zip(grads, model_params)) # zip (Gradient 계산한 것 , 업데이트할 파라미터) --- 즉, 계산할 걸 그 변수에 적용하는 코드

def plot(x, y):
    plt.plot(x, y)
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.show()

if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]  # 4 : Position , Velocity , Angle , Angular Velocity
    action_size = env.action_space.n             # 2 : Left / Right

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 400
    for e in range(num_episode):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        # print(state) [-0.03799004 -0.03124815 -0.03203197  0.04771467]
        # sys.exit()
        state = np.reshape(state, [1, state_size]) # 한 차원 높이기
        # print(state) [[-0.01093523 -0.02384852 -0.0402001  -0.03302338]]
        # sys.exit()

        while not done:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
            score += reward
            # Cart가 임계영역 바깥으로 빠져나가지 않거나, Pole이 임계 각도 내에 있다면 done = False
            reward = reward if not done or score == 500 else -1

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | memory length: {:4d} | X : {:.4f}m | epsilon: {:.4f}".format(
                      e, score_avg, len(agent.memory), info['X'], agent.epsilon))

                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("./save_graph/graph.png")

                if score_avg > 800:
                    agent.model.save_weights("./model", save_format="tf")
                    plot(episodes, scores)
                    sys.exit()

