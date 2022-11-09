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


env_config = bouncing_sprites.get_config(num_sprites=4,is_demo=False,visual_obs=True, add_sprite_info=True,
                                         discrete_all_sprite_mover=True,timeout_steps=50)

env = environment.Environment(**env_config)

gym_env = gym_wrapper.GymWrapper(env)


for i in range(10):
    obs = gym_env.reset()
    done = False
    while not done:
        action = gym_env.action_space.sample()
        obs, reward, done, info = gym_env.step(action)
        print(obs.shape)
        print(info)
        break
        if done:
            break



