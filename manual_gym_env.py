"""
"""
import numpy as np
import argparse
from moog import environment
from moog.env_wrappers import gym_wrapper
from moog_demos.example_configs import bouncing_sprites


def unroll_env(env):

    target_positions = [[0.25,0.25],[0.75,0.25],[0.25,0.75],[0.75,0.75]]
    print("Target positions", target_positions)
    total_reward = 0
    max_steps = 100
    state = env.reset()
    for t in range(max_steps):
        
        action = np.array([state[0],state[1], target_positions[0][0] , target_positions[0][1]  ])
        new_state, reward, done, info = env.step(action)
        print(state, reward)
        print(env._env._state)
        total_reward += reward
        state = new_state

        if done:
            print("total reward when ended",total_reward)
            break

def main(config):
    """Run interactive task demo."""

    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],is_demo=False,timeout_steps=50,sparse_rewards=config["sparse_rewards"])

    env = environment.Environment(**env_config)

    gym_env = gym_wrapper.GymWrapper(env)
    unroll_env(gym_env)
    unroll_env(gym_env)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sprites', type=int, default=1)
    parser.add_argument('--sparse_rewards', action="store_true")

    args = parser.parse_args()

    main(vars(args))


