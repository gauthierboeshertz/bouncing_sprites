"""
"""

import sys
import argparse
import importlib
import os
import numpy as np

from moog import env_wrappers
from moog import observers
from moog import environment
from moog.env_wrappers import gym_wrapper
from moog_demos.example_configs import bouncing_sprites
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed


def do_env_episode(env,model,episode_timesteps,save_gif_file=None):
    
    ep_reward = 0
    obs = env.reset()
    print(obs)
    for i in range(episode_timesteps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        ep_reward += rewards
        env.render()
        if done :
            if  i < (episode_timesteps -1 ):
                print("ENV IS SOLVED")
            break
    if save_gif_file:
        env.save_episode_gif(save_gif_file)
    return ep_reward

def main(config):

    
    print("CONFIG",config)

    set_random_seed(config["seed"])

    episode_timesteps = 100
    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],is_demo=False,timeout_steps=episode_timesteps, 
                                                                sparse_rewards=config["sparse_rewards"],
                                                                one_sprite_mover=config["one_sprite_mover"],
                                                                all_sprite_mover=config["all_sprite_mover"],
                                                                random_init_places=config["random_init_places"])


    env = environment.Environment(**env_config)

    gym_env = gym_wrapper.GymWrapper(env)
    
    print(gym_env.reset())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sprites', type=int, default=1)
    parser.add_argument('--sparse_rewards', action="store_true")
    parser.add_argument('--one_sprite_mover', action="store_true")
    parser.add_argument('--random_init_places', action="store_true")
    parser.add_argument('--all_sprite_mover', action="store_true")
    parser.add_argument('--algo', type=str,default="TD3")
    parser.add_argument('--learn_timesteps', type=int,default=20000)
    parser.add_argument('--seed', type=int,default=0)

    args = parser.parse_args()

    main(vars(args))


