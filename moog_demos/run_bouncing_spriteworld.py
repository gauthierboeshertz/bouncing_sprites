"""Demo script.

This script can be used to play with and test prototype tasks with a matplotlib
interface. This demo does not run with a fast frame-rate or control timing well,
so is not intended for colleting high-fidelity behavior (use a psychophysics
toolbox like MWorks for that), but is intended instead to sanity-check task
prototypes.

Run with the following:
$ python3 run_demo.py --config=path.to.your.config

See also the flags at the top of this file. There are options for rendering
settings, recording to gif, and logging the behavior.

To exit the demo on a mac, press the 'esc' key. To customize key bindings, see
the key bindings in human_agent.py. If you are playing a task with an action
space that is not supported by the interfaces in gui_frames.py, please add a
custom gui interface.
"""

import sys
sys.path.insert(0, '..')

import argparse
import os

from moog import env_wrappers
from moog import observers
from moog import environment
from moog_demos import gif_writer as gif_writer_lib
from moog_demos import human_agent
from moog_demos.example_configs.bouncing_sprites import get_config



def main(config):
    """Run interactive task demo."""
    render_size = 256
    print(config)
    env_config = get_config(num_sprites=config["num_sprites"],is_demo=True,random_init_places=config["random_init_places"],one_sprite_mover=config["one_sprite_mover"],sparse_reward=config["sparse_reward"],contact_reward=config["contact_reward"])

    print(env_config)
    env_config['observers']['image'] = observers.PILRenderer(
        image_size=(render_size, render_size),
        anti_aliasing=1,
        color_to_rgb=env_config['observers']['image'].color_to_rgb,
        polygon_modifier=env_config['observers']['image'].polygon_modifier,
        bg_color = (0,0,0)
    )

    if 'agents' in env_config:
        # Multi-agent demo
        agents = env_config.pop('agents')
        agent_name = env_config.pop('agent_name')
        multi_env = environment.Environment(**env_config)
        env = env_wrappers.MultiAgentEnvironment(
            environment=multi_env, agent_name=agent_name, **agents)
    else:
        # Single-agent demo
        env = environment.Environment(**env_config)

    if config["log_data"]:
        log_dir = os.path.join(
            'logs', "bouncing_spriteworld")
        env = env_wrappers.LoggingEnvironment(env, log_dir=log_dir)

    if config["write_gif"]:
        gif_writer = gif_writer_lib.GifWriter(
            gif_file="demo_gif",
            fps=30,
        )
    else:
        gif_writer = None

    # Constructing the agent automatically starts the environment
    human_agent.HumanAgent(
        env,
        render_size=render_size,
        fps=30,
        reward_history=100,
        gif_writer=gif_writer,
    )

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
    parser.add_argument('--write_gif', action="store_true")
    parser.add_argument('--log_data', action="store_true")

    args = parser.parse_args()

    main(vars(args))
