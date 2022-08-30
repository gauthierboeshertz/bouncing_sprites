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
from stable_baselines3 import A2C

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure


def main(config):


    print("CONFIG",config)
    episode_timesteps = 100
    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],is_demo=False,timeout_steps=episode_timesteps, 
                                                                sparse_rewards=config["sparse_rewards"],
                                                                one_sprite_mover=config["one_sprite_mover"],
                                                                all_sprite_mover=config["all_sprite_mover"],
                                                                random_init_places=config["random_init_places"])


    env = environment.Environment(**env_config)

    gym_env = gym_wrapper.GymWrapper(env)
    
    #check_env(gym_env)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    eval_callback = EvalCallback(gym_env, callback_on_new_best=callback_on_best, verbose=1)


    tensorboard_folder = "./logs/{}_{}sprites_{}_{}_{}".format(config["algo"],str(config["num_sprites"]),"sparse_rewards" if config["sparse_rewards"] else "", "one_sprite_mover" if config["one_sprite_mover"] else "all_sprite_mover" if config["one_sprite_mover"] else "select_move", "random_init" if config["random_init_places"] else "fixed_init")

    new_logger = configure(tensorboard_folder+"_log", ["stdout", "csv", "tensorboard"])

    model = eval(config["algo"])("MlpPolicy", gym_env, verbose=1,tensorboard_log=tensorboard_folder)
    model.set_logger(new_logger)

    model.learn(total_timesteps=20000,callback=eval_callback)

    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],is_demo=True,timeout_steps=episode_timesteps,
                                                                sparse_rewards=config["sparse_rewards"],
                                                                one_sprite_mover=config["one_sprite_mover"],
                                                                all_sprite_mover=config["all_sprite_mover"],
                                                                random_init_places=config["random_init_places"])

    env = environment.Environment(**env_config)

    gym_env = gym_wrapper.GymWrapper(env)

    obs = gym_env.reset()
    for i in range(episode_timesteps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = gym_env.step(action)
        gym_env.render()
        if done :
            if  i < (episode_timesteps -1 ):
                print("SIMULATION IS DONE")
            break
            
    gym_env.save_episode_gif("test_bounces.gif")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sprites', type=int, default=1)
    parser.add_argument('--sparse_rewards', action="store_true")
    parser.add_argument('--one_sprite_mover', action="store_true")
    parser.add_argument('--random_init_places', action="store_true")
    parser.add_argument('--all_sprite_mover', action="store_true")
    parser.add_argument('--algo', type=str,default="TD3")

    args = parser.parse_args()

    main(vars(args))


