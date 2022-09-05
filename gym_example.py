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
                                                                sparse_reward=config["sparse_reward"],
                                                                contact_reward=config["contact_reward"],
                                                                one_sprite_mover=config["one_sprite_mover"],
                                                                all_sprite_mover=config["all_sprite_mover"],
                                                                random_init_places=config["random_init_places"])


    env = environment.Environment(**env_config)

    gym_env = gym_wrapper.GymWrapper(env)
    
    #check_env(gym_env)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    eval_callback = EvalCallback(gym_env, callback_on_new_best=callback_on_best, verbose=1)


    tensorboard_folder = "./logs/{}/{}_{}sprites_{}_{}".format("sparse_rewards" if config["sparse_rewards"] else "l2_reward",config["algo"],str(config["num_sprites"]), "one_sprite_mover" if config["one_sprite_mover"] else "all_sprite_mover" if config["all_sprite_mover"] else "select_move", "random_init" if config["random_init_places"] else "fixed_init")

    new_logger = configure(tensorboard_folder+"_log", ["stdout", "csv", "tensorboard"])

    model = eval(config["algo"])("MlpPolicy", gym_env, verbose=0,seed=config["seed"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=config["learn_timesteps"],callback=eval_callback,n_eval_episodes=50,log_interval=config["learn_timesteps"])

    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],is_demo=True,timeout_steps=episode_timesteps,
                                                                sparse_rewards=config["sparse_rewards"],
                                                                one_sprite_mover=config["one_sprite_mover"],
                                                                all_sprite_mover=config["all_sprite_mover"],
                                                                random_init_places=config["random_init_places"])

    env = environment.Environment(**env_config)

    gym_env = gym_wrapper.GymWrapper(env)

    test_rewards = []
    num_tests = 10
    for i in range(num_tests):
        test_rewards.append(do_env_episode(gym_env,model,100,tensorboard_folder+".gif" if i == num_tests-1 else None))
    print("rewards for tests",test_rewards)
    print("mean reward",np.mean(test_rewards))
    print("reward std",np.std(test_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sprites', type=int, default=1)
    parser.add_argument('--contact_reward', action="store_true")
    parser.add_argument('--sparse_reward', action="store_true")
    parser.add_argument('--one_sprite_mover', action="store_true")
    parser.add_argument('--random_init_places', action="store_true")
    parser.add_argument('--all_sprite_mover', action="store_true")
    parser.add_argument('--algo', type=str,default="TD3")
    parser.add_argument('--learn_timesteps', type=int,default=20000)
    parser.add_argument('--seed', type=int,default=0)

    args = parser.parse_args()

    main(vars(args))


