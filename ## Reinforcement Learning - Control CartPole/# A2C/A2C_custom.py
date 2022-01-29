import sys
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform


class A2C(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C,self).__init__()

        self.actor_fc = Dense(24, activation = 'tanh')
        # 출력이 행동이기 때문에 Softmax activation 함수 사용
        self.actor_out = Dense(action_size, activation='softmax', kernel_initializer=RandomUniform(-1e-3, 1e-3))  # 가중치 초기화


        # 가치 신경망 정의 -- relu 보다 tanh가 더 성능이 좋았다.
        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):

        # 정책 신경망을 통해 Policy Improvement 수행
        actor_x = self.actor_fc(x)
        policy = self.actor_out(actor_x) # 출력 : Policy

        # 가치 신경망을 통해 Policy Evaluation을 수행 --- Delta function Update!
        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x) # 출력 : 가치 함수

        return policy, value

class A2CAgent:
    def __init__(self, action_size):
        self.render = True

        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망 가치신경망 생성
        self.model = A2C(self.action_size)

        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        # Clipnorm :
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=5.0)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):

        policy, _ = self.model(state) # 정책 신경망의 출력값
        policy = np.array(policy[0])
        return np.random.choice(self.action_size, 1, p=policy)[0] # Policy에 따라 Action을 선택

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):

        model_params = self.model.trainable_variables # Update할 파라미터들 가져오기
        with tf.GradientTape() as tape:
            policy, value = self.model(state)  # 정책 신경망과 가치 신경망의 출력 V(s) 가져오기
            _, next_value = self.model(next_state) # 모델에 다음 Step의 상태를 넣어 V(s+1) 가져오기

            # Advantage function 계산 delta = R + gamma*V(S_t+1) - V(S_t)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            one_hot_action = tf.one_hot([action], self.action_size)
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1) # 해당하는 행동의 확률 Get
            cross_entropy = - tf.math.log(action_prob + 1e-5)

            # Advantage function 계산 delta = R + gamma*V(S_t+1) - V(S_t)
            advantage = tf.stop_gradient(target - value[0]) # 정책신경망의 오류함수를 구하는 과정에서 가치 함수를 업데이트 하지 않기 위함!
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기
            loss = 0.2 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)

if __name__ == "__main__":

    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')

    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0
        loss_list = []
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # 매 타임스텝마다 학습
            loss = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | X : {:.4f}m  ".format(
                      e, score_avg, info['X']))

                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph_0.1.png")

                # 이동 평균이 400 이상일 때 종료
                if score_avg > 600:
                    agent.model.save_weights("./model", save_format="tf")
                    sys.exit()