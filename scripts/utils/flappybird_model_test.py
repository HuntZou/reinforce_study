import itertools
import time

import gymnasium
import torch
import flappy_bird_gymnasium
from loguru import logger


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(in_features=12, out_features=20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=20, out_features=20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=20, out_features=2),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.dnn(x)


actor = Actor()
actor.load_state_dict(torch.load(r"../../models/flappybird/adam0001.pt"))

test_env = gymnasium.make("FlappyBird-v0", audio_on=False)
for game_count in itertools.count():
    obs, _ = test_env.reset()
    rewards, acts = [], []
    for _ in itertools.count():
        probs = actor(torch.FloatTensor(obs))
        action = torch.argmax(probs).item()
        obs, reward, terminated, _, info = test_env.step(action)
        if obs[-3] < 0: reward, terminated = -1, True
        acts.append(action)
        rewards.append(reward)
        test_env.render()
        time.sleep(1 / 30)
        if terminated:
            logger.info(f"total reward: {sum(rewards):.4f}, fly_rate: {sum(acts) / len(acts):.4f}")
            break
test_env.close()
