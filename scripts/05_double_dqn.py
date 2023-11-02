"""
最原始的qlearning方法玩flappybird，使用两个model（并非DQN），使得训练更加稳定，每过几个epoch就同步一下参数
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

board = SummaryWriter('../log/ddqn')
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
            torch.nn.Linear(in_features=32, out_features=2),
        )

    def forward(self, state):
        return self.dnn(state)


model = QtableModel()
target_model = QtableModel()
target_model.load_state_dict(model.state_dict())
target_model.eval()

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

        # 与double model的方式不同点在于，ddqn使用正在训练的模型计算下一个状态最优的action（而double model是直接计算下一个状态的q value）
        # 然后使用该action传入到target model计算其qvalue，从而缓解qvalue被高估的问题
        next_state_q_val = target_model(torch.FloatTensor(np.array([next_state]))).gather(1, model(torch.FloatTensor(np.array([next_state]))).argmax(dim=1).detach().unsqueeze(-1))
        expect = torch.tensor([reward]).float() if terminate or truncated else reward + gamma * next_state_q_val.squeeze(-1)

        real = q_values.gather(1, torch.tensor([[action]])).squeeze(-1)

        optimizer.zero_grad()
        loss = loss_fn(real, expect)
        loss.backward()
        optimizer.step()

        state = next_state

        if terminate or truncated:
            if game_count % 100 == 0:
                target_model.load_state_dict(model.state_dict())
                logger.info(f'train count: {game_count}')
            break
