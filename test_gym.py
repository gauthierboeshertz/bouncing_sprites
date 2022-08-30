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
from stable_baselines3 import HER

from stable_baselines3.common.env_checker import check_env


env_config = bouncing_sprites.get_config(num_sprites=2,is_demo=False,timeout_steps=50,sparse_rewards=False)

env = environment.Environment(**env_config)

gym_env = gym_wrapper.GymWrapper(env)

check_env(gym_env)
model = HER("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=30000)



