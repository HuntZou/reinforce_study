# DDPG添加了一些优化，具体有：经验回放、批量训练、double net、延迟更新actor
# 该模型训练一个自动到停车位停车的模型，可以收敛，大概一个小时
import argparse
import collections
import itertools
import pathlib
import random
import threading
import time

import gymnasium as gym
import loguru
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from scripts.utils.utils import parking_state_to_tensor, visualize_parking_train

parser = argparse.ArgumentParser()
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
# gamma较小的情况下，模型难以收敛
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(13)
np.random.seed(13)

directory = r'./testlog/ddpg_parking_full_4'


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u], 1)))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = collections.deque(maxlen=1000000)
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = itertools.count()
        self.num_actor_update_iteration = itertools.count()

    def select_action(self, state):
        return self.actor(parking_state_to_tensor(state).to(device)).detach()

    def replay_data_to_tensor(self, datas):
        states, actions, rewards, next_states, dones = zip(*datas)

        states = torch.vstack([parking_state_to_tensor(state) for state in states]).float().to(device)
        actions = torch.vstack(actions).float().to(device)
        rewards = torch.tensor(rewards).unsqueeze(-1).float().to(device)
        next_states = torch.vstack([parking_state_to_tensor(state) for state in next_states]).float().to(device)
        dones = 1 - torch.FloatTensor(dones).unsqueeze(-1).float().to(device)

        return states, actions, rewards, next_states, dones

    def train(self):
        # SummaryWriter has pickle issue in multiprocess, so init it in every method
        while True:
            if len(self.replay_buffer) < 1000:
                loguru.logger.info(f"waiting for data collecting: {len(self.replay_buffer)}")
                time.sleep(5)
                continue

            states, actions, rewards, next_states, dones = self.replay_data_to_tensor(random.sample(self.replay_buffer, 1000))

            # Compute the target Q value
            target_Q = self.critic_target(next_states, self.actor_target(next_states))
            target_Q = rewards + (dones * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(states, actions)

            # Compute critic loss
            critic_loss = torch.nn.functional.mse_loss(current_Q, target_Q)
            critic_update_count = next(self.num_critic_update_iteration)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=critic_update_count)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if not (critic_update_count + 1) % 2:
                # Compute actor loss
                actor_loss = -self.critic(states, self.actor(states)).mean()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=next(self.num_actor_update_iteration))

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # print(f'critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    def collect_data(self):

        train_counter = itertools.count()

        env = gym.make('parking-v0')
        for game_count in itertools.count():
            total_reward = 0
            state, _ = env.reset()
            for step in itertools.count():
                action = self.select_action(state)
                if game_count < 300:
                    action = torch.rand(size=action.shape).to(device)
                rand_act = (action.cpu().squeeze(0) + np.random.normal(0, 0.2, size=env.action_space.shape[0])).clip(-1, 1)
                next_state, reward, done, truncate, info = env.step(rand_act)
                total_reward += reward
                if total_reward < -80: reward, truncate = -1, True

                self.replay_buffer.append((state, action, reward, next_state, done or truncate))

                state = next_state

                if done or truncate:
                    if not game_count % 10:
                        loguru.logger.info(f'game count: {game_count}, step: {step}, data_size: {len(self.replay_buffer)}')
                        torch.save(self.actor.state_dict(), r'./models/parking_actor1.pt')
                        torch.save(self.critic.state_dict(), r'./models/parking_critic1.pt')
                    break
            self.writer.add_scalar('train/total_reward', total_reward, next(train_counter))
            time.sleep(1)


def main():
    state_dim = 18
    action_dim = 2
    agent = DDPG(state_dim, action_dim)

    threading.Thread(target=agent.collect_data).start()
    threading.Thread(target=agent.train).start()
    threading.Thread(target=visualize_parking_train, args=(agent.actor, pathlib.Path(directory).absolute(), device)).start()


if __name__ == '__main__':
    main()
