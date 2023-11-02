# 在DDPG的基础上变成了 1. 模型输出概率分布而不是直接输出action  2. reward的gt增加action的熵值用于鼓励探索
import argparse
import collections
import itertools
import os.path
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
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
batch_size = 1000

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

        # self.action_scales = torch.Tensor([5., 0.785]).to(device)
        self.action_scales = torch.Tensor([1., 1.]).to(device)

    def forward2(self, x):
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

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.std(x)
        log_std = torch.tanh(log_std)
        log_std = -5 + 0.5 * 7 * (log_std + 1)  # From SpinUp / Denis Yarats

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scales

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scales * (1 - y_t.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


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
        self.replay_buffer = collections.deque(maxlen=batch_size * 100)
        self.writer = SummaryWriter(directory)

        self.log_ent_coef = torch.log(torch.ones(2, device=device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=1e-3)

    def select_action(self, state):
        return self.actor(parking_state_to_tensor(state).to(device))[0].detach()

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
        for train_count in itertools.count():
            if len(self.replay_buffer) < batch_size:
                loguru.logger.info(f"waiting for data collecting: {len(self.replay_buffer)}")
                time.sleep(5)
                continue

            states, actions, rewards, next_states, dones = self.replay_data_to_tensor(random.sample(self.replay_buffer, batch_size))

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
            target_Q = rewards + (dones * args.gamma * target_Q).detach()

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

                self.writer.add_scalar('Ent/ent_coef', ent_coef.cpu().detach().mean(), global_step=train_count)
                self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=train_count)
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=train_count)

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

                self.replay_buffer.append((state, action, reward, next_state, done or truncate))

                state = next_state

                if done or truncate:
                    if not game_count % 10:
                        loguru.logger.info(f'game count: {game_count}, step: {step}, data_size: {len(self.replay_buffer)}')
                        # torch.save(self.actor.state_dict(), r'./models/parking_actor1.pt')
                        # torch.save(self.critic.state_dict(), r'./models/parking_critic1.pt')
                    break
            self.writer.add_scalar('train/total_reward', total_reward, next(train_counter))


def main():
    state_dim = 18
    action_dim = 2
    agent = SAC(state_dim, action_dim)

    # agent.actor.load_state_dict(torch.load(r'./models/parking_actor1.pt'))
    # threading.Thread(target=visualize_parking_train, args=(agent.actor, None, device)).start()

    threading.Thread(target=agent.collect_data).start()
    threading.Thread(target=agent.train).start()
    threading.Thread(target=visualize_parking_train, args=(agent.actor, agent.writer, device)).start()


if __name__ == '__main__':
    main()
