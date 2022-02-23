# Emsemble Method를 추가해보자

# Method 1
# Hyperparemeter 동일
# 학습 모델 3개

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"On {device}")
print("")

def train(mu1,mu2,mu3, mu1_target,mu2_target, mu3_target, q1, q2, q3, q1_target, q2_target, q3_target, memory, q_optimizer, mu_optimizer):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    Q_loss, mu_loss = 0, 0

    y1 = rewards + (gamma * q1_target(next_states, mu1_target(next_states))) * dones
    y2 = rewards + (gamma * q2_target(next_states, mu2_target(next_states))) * dones
    y3 = rewards + (gamma * q3_target(next_states, mu3_target(next_states))) * dones

    result = torch.stack([y1, y2, y3], axis = 0)
    soft_result = torch.nn.functional.softmax(result, dim = 0) # Q value에 따라 가중치
    y_result = (result * soft_result)
    y1_weight = y_result[0]
    y2_weight = y_result[1]
    y3_weight = y_result[2]
    # y_stack = torch.cat([y1_weight,y2_weight,y3_weight], axis = 0)

    Q_loss1 = torch.nn.functional.smooth_l1_loss(q1(states, actions), y1_weight.detach())
    Q_loss2 = torch.nn.functional.smooth_l1_loss(q2(states, actions), y2_weight.detach())
    Q_loss3 = torch.nn.functional.smooth_l1_loss(q3(states, actions), y3_weight.detach())

    Q_loss = Q_loss1 + Q_loss2 + Q_loss3
    q_optimizer.zero_grad()
    Q_loss.backward()
    q_optimizer.step()

    mu_loss1 = -q1(states, mu1(states)).mean()
    mu_loss2 = -q2(states, mu2(states)).mean()
    mu_loss3 = -q3(states, mu3(states)).mean()
    mu_loss = mu_loss1 + mu_loss2 + mu_loss3
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

env = gym.make('Pendulum-v1')
memory = ReplayBuffer()

###################################################### Hyperparameters
lr_mu = 0.0005        # Learning Rate for Torque (Action)
lr_q = 0.001          # Learning Rate for Q

gamma = 0.99          # discount factor
batch_size = 16       # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 50000  # Replay Memory Size
tau = 0.005           # for target network soft update

###################################################### Models
q1  = QNet1().to(device)
q2 =  QNet2().to(device)
q3 =  QNet3().to(device)
q1_target = QNet1().to(device)
q2_target = QNet2().to(device)
q3_target = QNet3().to(device)

mu1 = MuNet1().to(device)
mu2 = MuNet2().to(device)
mu3 = MuNet3().to(device)
mu1_target = MuNet1().to(device)
mu2_target = MuNet2().to(device)
mu3_target = MuNet3().to(device)

# Parameter Synchronize
q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())
q3_target.load_state_dict(q3.state_dict())
mu1_target.load_state_dict(mu1.state_dict())
mu2_target.load_state_dict(mu2.state_dict())
mu3_target.load_state_dict(mu3.state_dict())

# Optimizer
mu_params = list(mu1.parameters()) + list(mu2.parameters()) + list(mu3.parameters())
q_params = list(q1.parameters()) + list(q2.parameters()) + list(q3.parameters())
mu_optimizer = optim.Adam(mu_params,  lr=lr_mu)
q_optimizer = optim.Adam(q_params, lr=lr_q)


#######################################################
score = 0.0
reward_history_20 = deque(maxlen=100)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
MAX_EPISODES = 500
# Action Space Map
A = np.arange(-2, 2, 0.001)
for episode in range(MAX_EPISODES):
    s = env.reset()
    done = False
    score = 0.0

    while not done: # Stacking Experiences

        # 3 개의 Model - 너도 그렇게 생각해 나도 그렇게 생각해 그런 느낌으로다가
        a1 = mu1(torch.from_numpy(s).float().to(device))  # Return action (-2 ~ 2 사이의 torque  ... )
        a2 = mu2(torch.from_numpy(s).float().to(device))  # Return action (-2 ~ 2 사이의 torque  ... )
        a3 = mu3(torch.from_numpy(s).float().to(device))  # Return action (-2 ~ 2 사이의 torque  ... )

        a = (a1+a2+a3) / 3
        # Discretize Action Space ...
        discrete_action = np.digitize(a.cpu().detach().numpy(), bins = A)

        # Soft Greedy
        sample = random.random()
        if sample < 0.1:
            random_action = np.array([random.randrange(0, len(A))])
            action = A[random_action - 1]

        else:
            action = A[discrete_action - 1]

        action = torch.from_numpy(action)
        print(action)
        print(action.shape)
        print(type(action))

        s_prime, r, done, info = env.step(action)
        memory.put((s, action, r / 100.0, s_prime, done))
        score = score + r
        s = s_prime

        if memory.size() > 2000:
            for i in range(10):
                train(mu1,mu2,mu3, mu1_target,mu2_target, mu3_target, q1, q2, q3, q1_target, q2_target, q3_target, memory, q_optimizer, mu_optimizer)
                soft_update(q1, q1_target)
                soft_update(q2, q2_target)
                soft_update(q3, q3_target)
                soft_update(mu1, mu1_target)
                soft_update(mu2, mu2_target)
                soft_update(mu3, mu3_target)


    reward_history_20.append(score)
    avg = sum(reward_history_20) / len(reward_history_20)
    if episode % 10 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, score, avg))
    episode = episode + 1

env.close()

# Record Hyperparamters & Result Graph
with open('DDPG_Discretization2.txt', 'w', encoding = 'UTF-8') as f:
    f.write("# ----------------------- # " + '\n')
    f.write("Parameter 2022-2-13" + '\n')
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

length = np.arange(len(reward_history_20))
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG_Discretization2")
plt.plot(length, reward_history_20)
plt.savefig('DDPG_Discretization2.png')
