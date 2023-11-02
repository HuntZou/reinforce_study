"""
基于动作-价值函数的actor-critic
"""
import itertools
import random
import threading

import gymnasium
import numpy as np
import torch
import flappy_bird_gymnasium
from loguru import logger

from scripts.utils.utils import visualize_FlappyBird_train, state_to_tensor
from torch.utils.tensorboard import SummaryWriter

board = SummaryWriter('./testlog/a2c_act_val')

gamma = 0.8

torch.random.manual_seed(0)
np.random.seed(0)
random.seed = 0


class Critic(torch.nn.Module):
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


class Actor(torch.nn.Module):
    """
    Policy model
    """

    def __init__(self):
        super().__init__()
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(in_features=12, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=32, out_features=2),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.dnn(x)


actor = Actor()
critic = Critic()

threading.Thread(target=visualize_FlappyBird_train, args=(actor, board), daemon=True).start()

critic_loss_fn = torch.nn.MSELoss()
actor_optimizer = torch.optim.Adam(params=actor.parameters(), lr=1e-3)
critic_optimizer = torch.optim.Adam(params=critic.parameters(), lr=1e-3)

env = gymnasium.make('FlappyBird-v0')

for game_count in itertools.count():
    state, _ = env.reset()
    while True:
        state = state_to_tensor(state)

        q_values = critic(state)
        probs = actor(state)

        action = torch.distributions.Categorical(probs).sample()[0].item() if random.random() < 0.98 else env.action_space.sample()

        next_state, reward, terminate, truncated, info = env.step(action)
        if next_state[-3] < 0: reward, truncated = -1, True

        # train critic-------
        # expect = torch.tensor([reward]).float() if terminate or truncated else reward + gamma * torch.max(critic(torch.FloatTensor(np.array([next_state]))), dim=1)[0].detach()
        # expect = r + gamma * next_stat_value - baseline
        with torch.no_grad():
            # expect = reward + gamma * critic(state_to_tensor(next_state)) - critic(state)
            # expect = reward + gamma * torch.sum(critic(state_to_tensor(next_state)) * actor(state_to_tensor(next_state)), dim=1) if not (terminate or truncated) else torch.FloatTensor([reward])  # 好像不收敛
            expect = reward + gamma * torch.max(critic(state_to_tensor(next_state)), dim=1)[0] if not (terminate or truncated) else torch.FloatTensor([reward])

        real = q_values.gather(1, torch.tensor([[action]])).squeeze(-1)

        critic_optimizer.zero_grad()
        critic_loss = critic_loss_fn(real, expect)
        critic_loss.backward()
        critic_optimizer.step()
        # end-----------------

        # train actor---------
        actor_optimizer.zero_grad()
        # real can be treated as baseline
        actor_loss = -(expect - 0).detach() * torch.log(probs + 1e-8).gather(1, torch.tensor([[action]]))
        actor_loss.backward()
        actor_optimizer.step()
        # end-----------------

        state = next_state
        if terminate or truncated:
            if game_count % 100 == 0:
                logger.info(f'train count: {game_count}')
            break

