"""
最原始的qlearning方法玩flappybird，除了固定final reward外，无任何优化技巧
"""
import collections
import itertools
import random
import threading

import gymnasium
import numpy as np
import torch
import flappy_bird_gymnasium
from loguru import logger

from scripts.utils.utils import visualize_FlappyBird_train
from torch.utils.tensorboard import SummaryWriter

board = SummaryWriter('../log/rainbow')

gamma = 0.8

torch.random.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QtableModel(torch.nn.Module):
    """
    Which can be replaced by a matrix
    """

    def __init__(self):
        super().__init__()
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(in_features=12, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
        )
        self.value_net = torch.nn.Linear(in_features=32, out_features=1)

        # 使用 layer norm 使得输出的概率和为0
        self.advantage_net = torch.nn.Linear(in_features=32, out_features=2)

    def forward(self, state):
        state = self.dnn(state)

        value = self.value_net(state)
        advantage = self.advantage_net(state)

        # advantage 减去均值，使得advantage的和为0，避免value_net学不到东西
        q_value = value + (advantage - advantage.mean(1).reshape(-1, 1))
        return q_value


model = QtableModel().to(device)
target_model = QtableModel().eval().to(device)
target_model.load_state_dict(model.state_dict())

# 实时可视化训练结果
threading.Thread(target=visualize_FlappyBird_train, args=(model, board, device), daemon=True).start()

def create_data(b: collections.deque):
    env = gymnasium.make('FlappyBird-v0')

    for game_count in itertools.count():
        state, info = env.reset()
        while True:

            q_values = model(torch.FloatTensor(np.array([state])).to(device))
            action = q_values.argmax(dim=1)[0].item() if np.random.random() < 0.3 else env.action_space.sample()

            next_state, reward, terminate, truncated, info = env.step(action)
            if state[-3] < 0: reward, terminate = -1, True

            b.append([state, action, reward, next_state, terminate or truncated])

            state = next_state

            if terminate or truncated:
                if game_count % 1000 == 0:
                    logger.info(f'train count: {game_count}')
                break


# 保存模型创造的数据
buffer = collections.deque(maxlen=1000)
threading.Thread(target=create_data, args=(buffer,), daemon=True).start()

BATCH_SIZE = 100
while len(buffer) < BATCH_SIZE: continue

# 主线程作为训练线程不断从buffer中取数据训练
loss_fn = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
for epoch in itertools.count():
    # 随机从buffer中获取100条数据，将它们分别组装成tensor
    states, actions, rewards, next_states, terminates = (np.array(i) for i in zip(*random.sample(buffer, BATCH_SIZE)))
    states, actions, rewards, next_states, terminates = torch.FloatTensor(states).to(device), \
        torch.tensor(actions, dtype=torch.int64).to(device), \
        torch.FloatTensor(rewards).to(device), \
        torch.FloatTensor(next_states).to(device), \
        torch.BoolTensor(terminates).to(device)

    next_states_q = target_model(next_states).gather(dim=1, index=torch.argmax(model(next_states), dim=1).detach().unsqueeze(-1)).squeeze(-1)
    expects = terminates * rewards + (~terminates) * (rewards + gamma * next_states_q)

    reals = model(states).gather(1, actions.unsqueeze(-1).detach()).squeeze(-1)
    loss = loss_fn(reals, expects)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # sync weight between eval model and target model every n step
    if epoch % 1000 == 0:
        logger.info(f'sync eval and target model weight at epoch: {epoch}')
        target_model.load_state_dict(model.state_dict())
