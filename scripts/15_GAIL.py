import itertools
import random

import gymnasium
import numpy as np
import torch
from loguru import logger

from utils.utils import Board

board = Board('./logs/gail_ppo_carpole')

torch.manual_seed(6)
np.random.seed(6)

device = torch.device('cuda:0')


# 模仿学习，基于PPO，大概两个小时收敛

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=32, out_features=2),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x):
        probs = self.dnn(x)
        return probs


class RewardNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(in_features=4 + 1, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, x):
        return self.dnn(x)


policy = Policy().to(device)
optim_policy = torch.optim.Adam(params=policy.parameters(), lr=0.001)

reward_net = RewardNet().to(device)
optim_reward = torch.optim.Adam(params=reward_net.parameters(), lr=0.001)


# 官方是每次训练收集 2048 * 8个环境并行收集 = 16384 条数据
def collect_data(max_len=1024 * 8):
    trajectories = []
    steps = []
    env = gymnasium.make('CartPole-v1')
    for game_count in itertools.count():
        trajectory = []
        state, _ = env.reset()
        for step in itertools.count():
            with torch.no_grad():
                probs = policy(torch.Tensor(state).to(device))
                dist = torch.distributions.Categorical(probs)
                act = dist.sample()
                log_prob = dist.log_prob(act)

                act, log_prob = act.cpu().item(), log_prob.detach().cpu()
                extra_reward = reward_net(torch.cat([torch.Tensor(state), torch.Tensor([act])], dim=0).to(device)).item()
            next_state, reward, done, trunc, info = env.step(act)

            trajectory.append([state, act, log_prob, reward, 0, extra_reward, 0, done or trunc, next_state])

            state = next_state
            if done or trunc:
                steps.append(step)

                base_line = np.average([i[4] for i in trajectories[-1024:]]) if trajectories else 0
                extra_base_line = np.average([i[6] for i in trajectories]) if trajectories else 0
                for s in range(step + 1):
                    trajectory[s][4] = sum([t[3] * 0.99 ** i for i, t in enumerate(trajectory[s:])]) - base_line  # advantage
                    trajectory[s][6] = sum([t[5] * 0.99 ** i for i, t in enumerate(trajectory[s:])]) - extra_base_line  # extra_advantage
                trajectories += trajectory
                break

        if len(trajectories) >= max_len:
            avg_reward_game, avg_reward_step = sum([t[3] for t in trajectories]) / (game_count + 1), sum([t[3] for t in trajectories]) / len(trajectories or [None])
            avg_extra_reward_game, avg_extra_reward_step = sum([t[5] for t in trajectories]) / (game_count + 1), sum([t[5] for t in trajectories]) / len(trajectories or [None])

            logger.info(f'Collect game count: {game_count:<5}, avg_steps: {sum(steps or [0]) // len(steps or [None]):<5}, avg_reward_game: {avg_reward_game:<8.2f}, avg_extra_reward_game: {avg_extra_reward_game:<8.2f}')

            board.add_scalar_map({'game_count': game_count,
                                  'avg_steps': sum(steps or [0]) / len(steps or [None]),
                                  'total_steps': sum(steps or [0]),
                                  'avg_reward_game': avg_reward_game,
                                  'avg_reward_step': avg_reward_step,
                                  'avg_extra_reward_game': avg_extra_reward_game,
                                  'avg_extra_reward_step': avg_extra_reward_step},
                                 "collect")
            break
    states, acts, log_probs, rewards, advantages, extra_rewards, extra_advantage, terminates, next_states = (np.array(i) for i in zip(*trajectories))
    states, actions, log_probs, rewards, advantages, extra_rewards, extra_advantage, terminates, next_states = torch.FloatTensor(states).to(device), \
        torch.tensor(acts, dtype=torch.int64).to(device), \
        torch.tensor(log_probs, dtype=torch.double).to(device), \
        torch.FloatTensor(rewards).to(device), \
        torch.FloatTensor(advantages).to(device), \
        torch.FloatTensor(extra_rewards).to(device), \
        torch.FloatTensor(extra_advantage).to(device), \
        torch.BoolTensor(terminates).to(device), \
        torch.FloatTensor(next_states).to(device),

    return states, actions, log_probs, rewards, advantages, extra_rewards, extra_advantage, next_states, terminates


# 官方是专家数据 64局 * 500步 = 32000 条数据
trajectories_expert = torch.load(r'../other/res/carpole_expert').to(device)


def train_reward(train_count=2, replay_buf=None):
    assert replay_buf is not None and len(replay_buf[0]) >= 1024
    indices = torch.randperm(len(replay_buf[0]))
    for e in range(train_count):
        for t in range(len(indices) // 1024):
            idxs = indices[t * 1024: (t + 1) * 1024]
            noobs = torch.cat([replay_buf[0][idxs], replay_buf[1][idxs].reshape([-1, 1])], dim=1)
            experts = trajectories_expert[random.sample(range(len(trajectories_expert)), 1024)][:, :-1]

            noob, expert = reward_net(noobs), reward_net(experts)

            board.add_scalars('disc/reward', {"exp": expert.mean().item(), "noob": noob.mean().item()})

            noob, expert = -torch.nn.functional.logsigmoid(-noob).mean(), -torch.nn.functional.logsigmoid(expert).mean()

            loss_reward = noob + expert
            optim_reward.zero_grad()
            loss_reward.backward()
            optim_reward.step()

            board.add_scalar('disc/loss', loss_reward.item())


def train_policy(train_count=5, replay_buf=None):
    assert replay_buf is not None and len(replay_buf[0]) >= 1024
    indices = torch.randperm(len(replay_buf[0]))
    for e in range(train_count):
        for t in range(len(indices) // 1024):
            idxs = indices[t * 1024: (t + 1) * 1024]
            states, actions, old_log_probs, rewards, advantages, extra_rewards, extra_advantage, next_states, terminates = [i[idxs] for i in replay_buf]

            probs = policy(states)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(actions)

            radio = torch.exp(new_log_probs - old_log_probs)
            clamp_radio = torch.clamp(torch.exp(new_log_probs - old_log_probs), 0.8, 1.2)
            loss_policy = -(torch.min(extra_advantage * radio, extra_advantage * clamp_radio)).sum()

            optim_policy.zero_grad()
            loss_policy.backward()
            optim_policy.step()

            board.add_scalar('policy/loss', loss_policy.item())


if __name__ == '__main__':
    for epoch in itertools.count():
        rb = collect_data(1024 * 8)
        train_reward(2, rb)
        train_policy(5, rb)

        # if epoch % 20 == 0:
        #     torch.save(policy.state_dict(), r'./outputs/policy_carpole.pth')
        #     torch.save(reward_net.state_dict(), r'./outputs/reward_carpole.pth')
