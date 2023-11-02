"""
不收敛
"""
import collections
import itertools
import threading

import gymnasium
import numpy as np
import numpy.random
import torch
import flappy_bird_gymnasium

from scripts.utils.utils import visualize_Pendulum_train
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

board = SummaryWriter('./testlog/discount_policy_gradient')

torch.manual_seed(0)
numpy.random.seed(0)


class Actor(torch.nn.Module):
    """
    Policy model
    """

    def __init__(self):
        super().__init__()
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
        )
        self.mean_net = torch.nn.Linear(in_features=32, out_features=1)
        self.deviate_net = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.dnn(x)
        mean = self.mean_net(x)
        deviate = self.deviate_net(x) + 1e-8
        return torch.distributions.normal.Normal(mean, deviate)


actor = Actor()

# new thread to visualize train process
threading.Thread(target=visualize_Pendulum_train, args=(actor, board), daemon=True).start()

# good lr are important
optimizer = torch.optim.Adam(params=actor.parameters(), lr=0.001)

env = gymnasium.make("Pendulum-v1")
discount_rewards_q = collections.deque(maxlen=100)  # used to calc baseline
for game_count in itertools.count():
    obs, _ = env.reset(seed=0)

    rewards, chosen_prob = [], []
    for s in itertools.count():
        distribution = actor(torch.FloatTensor(np.array([obs])))
        action = distribution.sample()

        obs, reward, terminate, truncated, info = env.step(action[0].detach().tolist())

        chosen_prob.append(distribution.log_prob(action))
        rewards.append(reward)

        if terminate or truncated:
            if game_count % 100 == 0:
                logger.info(f'train count: {game_count}')

            # apply reward discount and baseline to steps
            discount_rewards = [sum([r * 0.99 ** j for j, r in enumerate(rewards[i:])]) for i in range(len(rewards))]
            discount_rewards_q.append(sum(discount_rewards) / len(discount_rewards))
            baseline = sum(discount_rewards_q) / len(discount_rewards_q)
            loss = -sum([prob * (weighted_r - 0) for prob, weighted_r in zip(chosen_prob, discount_rewards)])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

env.close()
