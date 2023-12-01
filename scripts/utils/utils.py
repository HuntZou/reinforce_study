import random

import numpy as np
import pygame.event
import torch
import gymnasium
import itertools
import time
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import collections


class Board:
    def __init__(self, path):
        self.board = SummaryWriter(path)
        self.global_steps = collections.defaultdict(int)

    def add_scalar(self, tag, scalar):
        self.board.add_scalar(tag, scalar, self.global_steps[tag])
        self.global_steps[tag] += 1

    def add_scalars(self, tag, scalar):
        self.board.add_scalars(tag, scalar, self.global_steps[tag])
        self.global_steps[tag] += 1

    def add_scalar_map(self, m: dict, prefix=""):
        for k, v in m.items():
            self.add_scalar(f'{prefix}/{k}' if prefix else k, v)


def visualize_FlappyBird_train(policy: torch.nn.Module, board: SummaryWriter = None, device=torch.device("cpu")):
    with torch.no_grad():
        test_env = gymnasium.make("FlappyBird-v0", audio_on=False)
        for game_count in itertools.count():
            obs, _ = test_env.reset(seed=0)
            rewards, actions = [], []
            for step in itertools.count():
                test_env.render()
                pygame.event.clear()
                probs = policy(torch.FloatTensor(np.array([obs])).to(device))
                # print(f'prob: {probs}, state: {np.array([obs])}')
                action = torch.argmax(probs, dim=1)[0].item()
                obs, reward, terminate, truncated, info = test_env.step(action)
                if obs[-3] < 0: reward, truncated = -1, True

                actions.append(action)
                rewards.append(reward)

                time.sleep(1 / 30)
                if terminate or truncated:
                    logger.info(f"total reward: {sum(rewards):.4f}, fly rate: {sum(actions) / len(actions):.4f}")
                    if board:
                        board.add_scalar('test/total_reward', sum(rewards), game_count)
                        board.add_scalar('test/fly_rate', sum(actions) / len(actions), game_count)
                        board.add_scalar('test/steps', step, game_count)
                    break
        test_env.close()


def visualize_BipedalWalker_train(policy: torch.nn.Module, board, device):
    with torch.no_grad():
        test_env = gymnasium.make("BipedalWalker-v3", render_mode="human", hardcore=False)
        for game_count in itertools.count():
            obs, _ = test_env.reset()
            rewards = []
            while True:
                action = policy(torch.FloatTensor(obs).to(device))
                obs, reward, terminate, truncate, info = test_env.step(action.detach().cpu().numpy())
                if sum(rewards) < -25: reward, truncate = -1, True
                # print(f'action: {action.detach()}, reward: {reward}, total rewards: {sum(rewards)}')
                rewards.append(reward)
                time.sleep(1 / 15)
                if terminate or truncate:
                    logger.info(f"total reward: {sum(rewards):.4f}")
                    if board: board.add_scalar('test/total_reward', sum(rewards), game_count)
                    break
        test_env.close()


def state_to_tensor(state):
    return torch.FloatTensor(np.array([state]))


def visualize_CarRacing_train(policy: torch.nn.Module, board, device):
    with torch.no_grad():
        test_env = gymnasium.make("CarRacing-v2", domain_randomize=True, render_mode="human", continuous=True)
        for game_count in itertools.count():
            obs, _ = test_env.reset(options={"randomize": False})
            rewards = []
            for step in itertools.count():
                action = policy(img_state_to_tensor(obs).to(device)).squeeze(0).cpu().numpy()
                # action = [random.random()*2-1, random.random(), random.random()]
                # action = [0, random.random(), 0]
                action = [0, -1, 0]
                obs, reward, terminated, truncated, info = test_env.step(action)
                rewards.append(reward)
                if sum(rewards) < 0: truncated, reward = True, -10
                # print(f'act: {action}, reward: {reward:.4f}, total reward: {sum(rewards):.4f}')
                time.sleep(1 / 15)
                if terminated or truncated:
                    logger.info(f"total reward: {sum(rewards):.4f}")
                    if board:
                        board.add_scalar('test/total_reward', sum(rewards), game_count)
                        board.add_scalar('test/steps', step, game_count)
                    break
        test_env.close()


def visualize_parking_train(policy: torch.nn.Module, board: SummaryWriter, device):
    """
    action为 [速度, 转向]。值域分别为 (-1, 1)，(-1, 1)
    """
    with torch.no_grad():
        test_env = gymnasium.make("parking-v0", render_mode="human")

        for game_count in itertools.count():
            obs, _ = test_env.reset(options={"randomize": False})
            rewards = []
            for step in itertools.count():
                # action = policy.get_act(parking_state_to_tensor(obs).to(device))[0].squeeze(0).cpu().numpy()
                pred = policy(parking_state_to_tensor(obs).to(device))
                if isinstance(pred, tuple):
                    action = pred[0].squeeze(0).cpu().numpy()
                else:
                    action = pred.squeeze(0).cpu().numpy()
                # print(action)
                obs, reward, terminated, truncated, info = test_env.step(action)
                rewards.append(reward)
                if sum(rewards) < -80: truncated = True
                # print(f'act: {action}, reward: {reward:.4f}, total reward: {sum(rewards):.4f}')
                time.sleep(1 / 30)
                if terminated or truncated:
                    logger.info(f"total reward: {sum(rewards):.4f}")
                    if board:
                        board.add_scalar('test/total_reward', sum(rewards), game_count)
                        board.add_scalar('test/steps', step, game_count)
                    break
        test_env.close()


def visualize_pendulum_train(policy: torch.nn.Module, board, device):
    with torch.no_grad():
        test_env = gymnasium.make("Pendulum-v1", render_mode="human")

        for game_count in itertools.count():
            obs, _ = test_env.reset()
            rewards = []
            for step in itertools.count():
                action = policy(state_to_tensor(obs).to(device)).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = test_env.step(action)
                rewards.append(reward)
                # print(f'act: {action}, reward: {reward:.4f}, total reward: {sum(rewards):.4f}')
                time.sleep(1 / 15)
                if terminated or truncated:
                    logger.info(f"total reward: {sum(rewards):.4f}")
                    if board:
                        board.add_scalar('test/total_reward', sum(rewards), game_count)
                        board.add_scalar('test/steps', step, game_count)
                    break
        test_env.close()


def visualize_pendulum_train2(policy: torch.nn.Module, board, device):
    with torch.no_grad():
        test_env = gymnasium.make("Pendulum-v1", render_mode="human")
        # test_env = gymnasium.vector.SyncVectorEnv([make_env("Pendulum-v1", 0, 0, False, "aaa")])

        for game_count in itertools.count():
            obs, _ = test_env.reset()
            rewards = []
            for step in itertools.count():
                action, _, _ = policy.get_action(torch.Tensor(obs).reshape([1, -1]).to(device))
                action = action.detach().reshape([1, ]).cpu().numpy()
                obs, reward, terminated, truncated, info = test_env.step(action)
                rewards.append(reward)
                # print(f'act: {action}, reward: {reward:.4f}, total reward: {sum(rewards):.4f}')
                time.sleep(1 / 15)
                if terminated or truncated:
                    logger.info(f"total reward: {sum(rewards):.4f}")
                    if board:
                        board.add_scalar('test/total_reward', sum(rewards), game_count)
                        board.add_scalar('test/steps', step, game_count)
                    break
        test_env.close()


def visualize_highway_train(policy: torch.nn.Module, board, device):
    with torch.no_grad():
        test_env = gymnasium.make("highway-v0", render_mode="human")
        test_env.configure({
            "action": {
                "type": "ContinuousAction",
            },
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
            }
        })

        for game_count in itertools.count():
            obs, _ = test_env.reset(options={"randomize": False})
            rewards = []
            for step in itertools.count():
                action = policy(highway_state_to_tensor(obs).to(device)).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = test_env.step(action)
                if len(rewards) > 8 and not sum(rewards[-8:]): reward, truncated = -1, True

                rewards.append(reward)
                if sum(rewards) < -80: truncated = True
                # print(f'act: {action}, reward: {reward:.4f}, total reward: {sum(rewards):.4f}')
                time.sleep(1 / 15)
                if terminated or truncated:
                    logger.info(f"total reward: {sum(rewards):.4f}")
                    if board:
                        board.add_scalar('test/total_reward', sum(rewards), game_count)
                        board.add_scalar('test/steps', step, game_count)
                    break
        test_env.close()


def visualize_landing_train(policy: torch.nn.Module, board, device):
    with torch.no_grad():
        test_env = gymnasium.make('LunarLander-v2', continuous=True, render_mode="human")

        for game_count in itertools.count():
            obs, _ = test_env.reset()
            rewards = []
            for step in itertools.count():
                action = policy(state_to_tensor(obs).to(device)).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = test_env.step(action)

                rewards.append(reward)
                # print(f'act: {action}, reward: {reward:.4f}, total reward: {sum(rewards):.4f}')
                time.sleep(1 / 15)
                if terminated or truncated:
                    logger.info(f"total reward: {sum(rewards):.4f}")
                    if board:
                        board.add_scalar('test/total_reward', sum(rewards), game_count)
                        board.add_scalar('test/steps', step, game_count)
                    break
        test_env.close()


def state_to_tensor(state):
    return torch.FloatTensor(np.array([state]))


def parking_state_to_tensor(state):
    return torch.tensor(np.array([*state.values()]), dtype=torch.float).reshape([1, -1])


def highway_state_to_tensor(state):
    return torch.FloatTensor(np.array(state).flatten())


def img_state_to_tensor(state):
    return torch.FloatTensor(np.array([state])).permute([0, 3, 1, 2])
