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

from absl import app
from absl import flags
import importlib
import os

from moog import env_wrappers
from moog import observers
from moog import environment
from moog_demos import gif_writer as gif_writer_lib
from moog_demos import human_agent
from moog_demos.example_configs.bouncing_sprites import get_config
FLAGS = flags.FLAGS
flags.DEFINE_string('config',
                    'moog_demos.example_configs.pong',
                    'Filename of task config to use.')
flags.DEFINE_integer('num_sprites', 1, 'Whether to log timestep data.')
flags.DEFINE_integer('render_size', 256,
                     'Height and width of the output image.')
flags.DEFINE_integer('anti_aliasing', 1, 'Renderer anti-aliasing factor.')
flags.DEFINE_integer('fps', 50,
                     'Upper bound on frames per second. Note: this is not an '
                     'accurate fps for the demo, since matplotlib and tkinter '
                     'introduce additional lag.')
flags.DEFINE_integer('reward_history', 30,
                     'Number of historical reward timesteps to plot.')

# Flags for gif writing
flags.DEFINE_boolean('write_gif', False, 'Whether to write a gif.')
flags.DEFINE_string('gif_file',
                    '/logs/gifs/test.gif',
                    'File path to write the gif to.')
flags.DEFINE_integer('gif_fps', 15, 'Frames per second for the gif.')

# Flags for logging timestep data
flags.DEFINE_boolean('log_data', False, 'Whether to log timestep data.')
flags.DEFINE_boolean('one_sprite_mover', False, 'Whether to log timestep data.')
flags.DEFINE_boolean('sparse_reward', True, 'use sparse reward which are only one if the sprite touches the goal.')
flags.DEFINE_boolean('random_init_places', False, 'Whether to log timestep data.')


def main(_):
    """Run interactive task demo."""
    print(FLAGS.one_sprite_mover)
    config = get_config(num_sprites=FLAGS.num_sprites,is_demo=True,random_init_places=FLAGS.random_init_places,one_sprite_mover=FLAGS.one_sprite_mover)

    print(config)
    config['observers']['image'] = observers.PILRenderer(
        image_size=(FLAGS.render_size, FLAGS.render_size),
        anti_aliasing=FLAGS.anti_aliasing,
        color_to_rgb=config['observers']['image'].color_to_rgb,
        polygon_modifier=config['observers']['image'].polygon_modifier,
    )

    if 'agents' in config:
        # Multi-agent demo
        agents = config.pop('agents')
        agent_name = config.pop('agent_name')
        multi_env = environment.Environment(**config)
        env = env_wrappers.MultiAgentEnvironment(
            environment=multi_env, agent_name=agent_name, **agents)
    else:
        # Single-agent demo
        env = environment.Environment(**config)

    if FLAGS.log_data:
        log_dir = os.path.join(
            'logs', FLAGS.config.split('.')[-1], str(FLAGS.level))
        env = env_wrappers.LoggingEnvironment(env, log_dir=log_dir)

    if FLAGS.write_gif:
        gif_writer = gif_writer_lib.GifWriter(
            gif_file=FLAGS.gif_file,
            fps=FLAGS.gif_fps,
        )
    else:
        gif_writer = None

    # Constructing the agent automatically starts the environment
    human_agent.HumanAgent(
        env,
        render_size=FLAGS.render_size,
        fps=FLAGS.fps,
        reward_history=FLAGS.reward_history,
        gif_writer=gif_writer,
    )


if __name__ == "__main__":
    app.run(main)
