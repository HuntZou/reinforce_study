"""
最原始的qlearning方法玩flappybird，除了固定final reward外，无任何优化技巧
"""
import itertools
import threading

import gymnasium
import numpy as np
import torch
import flappy_bird_gymnasium
from loguru import logger

from scripts.utils.utils import visualize_FlappyBird_train
from torch.utils.tensorboard import SummaryWriter

board = SummaryWriter('../log/dueling_dqn')
gamma = 0.8

torch.random.manual_seed(0)
np.random.seed(0)


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


model = QtableModel()

threading.Thread(target=visualize_FlappyBird_train, args=(model, board), daemon=True).start()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

env = gymnasium.make('FlappyBird-v0')

for game_count in itertools.count():
    state, _ = env.reset()
    while True:

        q_values = model(torch.FloatTensor(np.array([state])))
        action = q_values.argmax(dim=1)[0].item() if np.random.random() < 0.3 else env.action_space.sample()

        next_state, reward, terminate, truncated, info = env.step(action)
        if state[-3] < 0: reward, terminate = -1, True

        # Note that, fix terminate reward is important, like recurse need edge cause. otherwise, model hard to convergence
        # expect = reward + gamma * torch.max(model(torch.FloatTensor([next_state]))).detach()
        expect = torch.tensor([reward]).float() if terminate or truncated else reward + gamma * torch.max(model(torch.FloatTensor(np.array([next_state]))), dim=1)[0].detach()

        real = q_values.gather(1, torch.tensor([[action]])).squeeze(-1)

        optimizer.zero_grad()
        loss = loss_fn(real, expect)
        loss.backward()
        optimizer.step()

        state = next_state

        if terminate or truncated:
            if game_count % 100 == 0:
                logger.info(f'train count: {game_count}')
            break
