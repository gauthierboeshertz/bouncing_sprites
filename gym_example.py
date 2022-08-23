"""
"""

import sys
import argparse
import importlib
import os

from moog import env_wrappers
from moog import observers
from moog import environment
from moog.env_wrappers import gym_wrapper
from moog_demos.example_configs import bouncing_sprites
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3 import SAC

def main(config):
    """Run interactive task demo."""

    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],is_demo=False,timeout_steps=50,sparse_rewards=config["sparse_rewards"])

    env = environment.Environment(**env_config)

    gym_env = gym_wrapper.GymWrapper(env)

    model = TD3("MlpPolicy", gym_env, verbose=1,learning_rate=0.0001)
    model.learn(total_timesteps=1000000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sprites', type=int, default=1)
    parser.add_argument('--sparse_rewards', action="store_true")

    args = parser.parse_args()

    main(vars(args))


