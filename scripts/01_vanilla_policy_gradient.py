"""
最原始的policy gradient算法玩flappybird，除了discount factor外，无任何优化技巧
"""
import collections
import itertools
import threading

import gymnasium
import numpy as np
import numpy.random
import torch
import flappy_bird_gymnasium

from scripts.utils.utils import visualize_FlappyBird_train
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

board = SummaryWriter('../log/vanilla_policy_gradient')

torch.manual_seed(0)
numpy.random.seed(0)


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

# new thread to visualize train process
threading.Thread(target=visualize_FlappyBird_train, args=(actor, board), daemon=True).start()

# good lr are important
optimizer = torch.optim.Adam(params=actor.parameters(), lr=0.001)

env = gymnasium.make("FlappyBird-v0")
discount_rewards_q = collections.deque(maxlen=100)  # used to calc baseline
for game_count in itertools.count():
    obs, _ = env.reset(seed=0)

    rewards, chosen_act_probs = [], []
    for s in itertools.count():
        action_probs = actor(torch.FloatTensor(np.array([obs])))
        # sample action from output distribution, and add some random action as epsilon greedy
        action = torch.distributions.Categorical(action_probs).sample().item() if np.random.random() < 0.98 else np.random.randint(0, 2)

        obs, reward, terminate, truncated, info = env.step(action)
        # hit ceiling: game over and punish
        if obs[-3] < 0: reward, terminate = -1, True

        # log is important
        chosen_act_probs.append(torch.log(action_probs + 1e-8).gather(1, torch.tensor([[action]])))
        rewards.append(reward)

        if terminate or truncated:
            if game_count % 100 == 0:
                logger.info(f'train count: {game_count}')
            # pply total reward to every steps in one trajectory
            loss = -sum(rewards) * sum(chosen_act_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

env.close()
