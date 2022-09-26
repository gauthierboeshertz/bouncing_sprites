"""
"""

import sys
import argparse
import importlib
import os
import numpy as np

from gym_example import run_gym_example

def run_seeds(num_seeds,config):

    seed_results = []
    for seed in range(num_seeds):
        config["seed"] = seed
        seed_results_, exp_name = run_gym_example(config)
        seed_results.append(seed_results_)
    
    seed_results = np.array(seed_results)
    np.save(exp_name+"_seed_results.npy",seed_results)
    print(seed_results)

def main(algo, num_seeds, do_l2=False,do_sparse=False,do_contact=False,start_sprites=1,end_sprites=5):

    config = {}
    config["all_sprite_mover"] = True
    config["learn_timesteps"] = 100000
    config["one_sprite_mover"] = False

    config["algo"] = algo
    for num_sprites in range(start_sprites,end_sprites):
        config["num_sprites"] = num_sprites

        for random_init_places in [True,False]:
            config["random_init_places"] = random_init_places

            if do_contact:
                config["contact_reward"] = True
                config["sparse_reward"] = False
                run_seeds(num_seeds,config)

            if do_sparse:
                config["contact_reward"] = False
                config["sparse_reward"] = True
                run_seeds(num_seeds,config)

            if do_l2:
                config["contact_reward"] = False
                config["sparse_reward"] = False
                run_seeds(num_seeds,config)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int,default=10)
    parser.add_argument('--algo', type=str,default="TD3")
    parser.add_argument('--do_l2', action="store_true")
    parser.add_argument('--do_sparse', action="store_true")
    parser.add_argument('--do_contact', action="store_true")
    parser.add_argument('--start_sprites', type=int,default=1)
    parser.add_argument('--end_sprites', type=int,default=5)

    args = parser.parse_args()

    main(args.algo,args.num_seeds,args.do_l2,args.do_sparse,args.do_contact,args.start_sprites,args.end_sprites)


