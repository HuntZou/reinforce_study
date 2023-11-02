"""
不收敛，DDPG一般用来处理连续动作空间问题
"""
import itertools
import threading
import time

import gymnasium
import numpy as np
import torch
from loguru import logger

from scripts.utils.utils import visualize_BipedalWalker_train
from torch.utils.tensorboard import SummaryWriter

board = SummaryWriter('./testlog/ddpg_', time.strftime('%y_%m_%d_%H_%M_%S'), time.localtime())

gamma = 0.8

torch.random.manual_seed(3)
np.random.seed(0)


class Critic(torch.nn.Module):
    """
    Which can be replaced by a matrix
    """

    def __init__(self):
        super().__init__()
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=24, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
        )

        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, state, action):
        x = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=1)
        return self.decoder(x)


class Actor(torch.nn.Module):
    """
    Policy model
    """

    def __init__(self):
        super().__init__()
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(in_features=24, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=32, out_features=4),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.dnn(x) * 2


actor = Actor()
critic = Critic()

threading.Thread(target=visualize_BipedalWalker_train, args=(actor, board), daemon=True).start()

critic_loss_fn = torch.nn.MSELoss()
actor_optimizer = torch.optim.Adam(params=actor.parameters(), lr=1e-3)
critic_optimizer = torch.optim.Adam(params=critic.parameters(), lr=1e-3)

env = gymnasium.make('BipedalWalker-v3', hardcore=False)

for game_count in itertools.count():
    state, _ = env.reset()
    while True:
        action = actor(torch.FloatTensor(np.array([state])))

        next_state, reward, terminate, truncated, info = env.step(action.detach().squeeze(0).tolist())

        # train critic-------
        expect = torch.tensor([[reward]]).float() if terminate or truncated else reward + gamma * critic(torch.FloatTensor(np.array([next_state])), actor(torch.FloatTensor(np.array([next_state])))).detach()
        real = critic(torch.FloatTensor(np.array([state])), action.detach())

        critic_optimizer.zero_grad()
        critic_loss = critic_loss_fn(real, expect)
        critic_loss.backward()
        critic_optimizer.step()
        # end-----------------

        # train actor---------
        # because I just do actor_optimizer.step(), so critic net will not update
        actor_loss = -critic(torch.FloatTensor(np.array([state])), action)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # end-----------------

        state = next_state
        if terminate or truncated:
            if game_count % 100 == 0:
                logger.info(f'train count: {game_count}')
            break
