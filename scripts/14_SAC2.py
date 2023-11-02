# 重写了 replay buffer 部分，使得训练速度更快，GPU利用率更高
import argparse
import itertools
import os.path
import threading
import time
from collections import OrderedDict
from typing import SupportsFloat

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
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
batch_size = 2048

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(13)
np.random.seed(13)

file_name = os.path.basename(__file__).split('.')[0] + "_" + str(time.time())[:10]
directory = rf'./testlog/{file_name}'


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.mean = nn.Linear(128, action_dim)
        self.std = nn.Linear(128, action_dim)

        self.action_scales = torch.Tensor([1., 1.]).to(device)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))

        mean = self.mean(x)
        std = self.std(x)

        std = torch.clamp(std, -20, 2)
        std = torch.exp(std)

        dist = torch.distributions.normal.Normal(mean, std)
        act_gus = dist.rsample()
        act = torch.tanh(act_gus)

        act = self.action_scales * act

        log_prob = dist.log_prob(act_gus) - torch.log(self.action_scales * (1 - act.pow(2)) + 1e-6)

        return act, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u], 1)))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


class ReplayBuf:
    def __init__(self, cap: int, device, action_dim: int = 2, observation_dim: int = 18):
        self._device = device

        self._states = torch.zeros(size=(cap, observation_dim)).to(self._device)
        self._actions = torch.zeros(size=(cap, action_dim)).to(self._device)

        self._next_states = torch.zeros(size=(cap, observation_dim)).to(self._device)
        self._rewards = torch.zeros(size=(cap, 1)).to(self._device)
        self._dones = torch.zeros(size=(cap, 1)).to(self._device)
        self._truncates = torch.zeros(size=(cap, 1)).to(self._device)

        self._cap = cap
        self._pos = 0

    def add(self, state: OrderedDict, action: torch.Tensor, next_state: OrderedDict, reward: SupportsFloat, done: bool, truncate: bool):
        insert_idx = self._pos % self._cap

        self._states[insert_idx] = torch.tensor(np.array([*state.values()]), dtype=torch.float).reshape([-1, ]).to(self._device)
        self._actions[insert_idx] = action.clone().detach().reshape([-1, ]).to(self._device)
        self._next_states[insert_idx] = torch.tensor(np.array([*next_state.values()]), dtype=torch.float).reshape([-1, ]).to(self._device)
        self._rewards[insert_idx] = torch.tensor(reward, dtype=torch.float).to(self._device)
        self._dones[insert_idx] = torch.tensor(done, dtype=torch.float).to(self._device)
        self._truncates[insert_idx] = torch.tensor(truncate, dtype=torch.float).to(self._device)

        self._pos += 1

    def sample(self, batch_size: int):
        idx = torch.randint(low=0, high=min([self._pos, self._cap]), size=(batch_size,))

        states, actions = self._states[idx], self._actions[idx]
        next_states, rewards = self._next_states[idx], self._rewards[idx],
        dones, truncates = self._dones[idx], self._truncates[idx]

        return states, actions, next_states, rewards, dones, truncates

    def __len__(self):
        return min([self._pos, self._cap])


class SAC(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.rb = ReplayBuf(cap=batch_size * 100, device=device, )

        self.writer = SummaryWriter(directory)

        self.log_ent_coef = torch.log(torch.ones(2, device=device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=1e-3)

    def select_action(self, state):
        return self.actor(parking_state_to_tensor(state).to(device))[0].detach()

    def train(self):
        # SummaryWriter has pickle issue in multiprocess, so init it in every method
        for train_count in itertools.count():
            if len(self.rb) < batch_size:
                loguru.logger.info(f"waiting for data collecting: {len(self.rb)} of {batch_size}")
                time.sleep(5)
                continue
            if train_count % 500 == 0:
                loguru.logger.info(f'train count: {train_count}')

            states, actions, next_states, rewards, dones, truncates = self.rb.sample(batch_size=batch_size)

            act, act_log_prob = self.actor(states)

            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (act_log_prob - 2).detach()).mean()  # -2 = np.prod(env.action_space.shape)
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            # Compute the target Q value
            act_next, log_prob_next = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, act_next) - (ent_coef * log_prob_next.detach()).sum(dim=1, keepdim=True)
            # target_Q = self.critic_target(next_states, act_next)
            target_Q = rewards + ((1 - ((dones + truncates) > 0).float()) * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(states, actions)

            # current_Q.register_hook(lambda g: print("grad: ", g))

            # Compute critic loss
            critic_loss = torch.nn.functional.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if not (train_count + 1) % 2:
                # Compute actor loss
                actor_loss = ((ent_coef * act_log_prob).sum(dim=1, keepdim=True) - self.critic(states, act)).mean()
                # actor_loss = (- self.critic(states, act)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # print(f'critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

                self.writer.add_scalar('Ent/log_prob', act_log_prob.cpu().detach().mean(), global_step=train_count)
                self.writer.add_scalar('Ent/ent_coef', ent_coef.cpu().detach().mean(), global_step=train_count)
                self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=train_count)
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=train_count)

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

                self.rb.add(state, action, next_state, reward, done, truncate)

                state = next_state

                if done or truncate:
                    if not game_count % 20:
                        loguru.logger.info(f'collect game count: {game_count}, step: {step}, data_size: {len(self.rb)} data_pos: {self.rb._pos}')
                        torch.save(self.actor.state_dict(), rf'./models/parking_actor_{file_name}.pt')
                        torch.save(self.critic.state_dict(), rf'./models/parking_critic_{file_name}.pt')
                    break
            self.writer.add_scalar('train/total_reward', total_reward, next(train_counter))


def main():
    state_dim = 18
    action_dim = 2
    agent = SAC(state_dim, action_dim)

    agent.actor.load_state_dict(torch.load(rf'./models/parking_actor_SAC_new_rb_1698885356.pt'))
    threading.Thread(target=visualize_parking_train, args=(agent.actor, None, device)).start()

    # threading.Thread(target=agent.collect_data).start()
    # threading.Thread(target=agent.train).start()
    # threading.Thread(target=visualize_parking_train, args=(agent.actor, agent.writer, device)).start()


if __name__ == '__main__':
    main()
