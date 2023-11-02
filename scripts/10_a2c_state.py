# 基于状态-价值函数的actor-critic，由于critic和actor都是同步训练的，可能导致critic对状态价值的高估或低估，进而导致advantage值与真实情况相反（例如本来应该是负值减小某action的概率，结果它输出正值反而增大了该action），这种情况下，模型有一定概率失效
# 一个解决方法是先让critic训练一段时间后再训练actor，实验表明，这种方法能有效提升模型性能。这种方法也被运用于后续的TD3模型中
import itertools
import time

import torch

torch.manual_seed(0)
from scripts.utils.utils import state_to_tensor, visualize_FlappyBird_train
import gymnasium
import flappy_bird_gymnasium
import threading
from torch.utils.tensorboard import SummaryWriter

board = SummaryWriter('./testlog/a2c_state_val')


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
            torch.nn.Linear(in_features=32, out_features=1),
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


gamma = 0.8

actor = Actor()
critic = Critic()

critic_loss_fn = torch.nn.MSELoss()
critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

threading.Thread(target=visualize_FlappyBird_train, args=(actor, board)).start()
env = gymnasium.make("FlappyBird-v0")
train_count = 0
for game_count in itertools.count():
    state, _ = env.reset()
    for step in itertools.count():
        train_count += 1
        action_probs = actor(state_to_tensor(state))
        value = critic(state_to_tensor(state))
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        next_state, reward, determine, truncate, info = env.step(action)
        if next_state[-3] < 0: reward, truncate = -1, True

        predict = float(reward) + gamma * critic(state_to_tensor(next_state)).detach()
        expect = predict if not (determine or truncate) else torch.FloatTensor([[reward]])

        critic_loss = critic_loss_fn(expect, value)
        board.add_scalar('train/critic_loss', critic_loss.item(), train_count)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        actor_loss = -(predict - value).detach() * action_log_prob
        board.add_scalar('train/actor_loss', actor_loss.item(), train_count)
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        state = next_state

        if determine or truncate:
            break
