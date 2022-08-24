"""Avoid colliding predator polygons.

This task serves to showcase collisions. The predators have a variety of
polygonal shapes and bounce off each other and off the walls with Newtonian
collisions. The subject controls a green agent circle. The subject gets negative
reward if contacted by a predators and positive reward periodically.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import physics as physics_lib
from moog import observers
from moog import sprite
from moog import tasks
from moog import shapes
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators

RAW_REWARD_MULTIPLIER = 5
TERMINATE_DISTANCE = 0.05
color_list = [[0,0,255],[255,0,0],[128,0,0],[255,255,0],[128,0,128]]
TARGET_POSITIONS = [[0.25,0.25],[0.75,0.25],[0.25,0.75],[0.75,0.75]]
def get_config(num_sprites,is_demo=True,timeout_steps=1000,sparse_rewards=True):
    """Get environment config."""

    print("Using bouncing ball environment with {} sprites".format(num_sprites))
    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent

    # Walls
    walls = shapes.border_walls(visible_thickness=0.05, c0=0, c1=0, c2=0)

    # Create callable initializer returning entire state

    def state_initializer():
        targets = []
        agents = []
        targets_overlap = None
        agents_overlap = None
        for i in range(num_sprites):
            agents_factors = distribs.Product(
                [distribs.Continuous('x', 0.1, 0.9),
                distribs.Continuous('y', 0.1, 0.9)],
                #[distribs.Discrete('x', [target_positions[i][0]-0.1]),
                #distribs.Discrete('y', [target_positions[i][1]-0.1])],

                shape='circle', scale=0.1, c0=color_list[i][0], c1=color_list[i][1], c2=color_list[i][2],
            )
            target_factors = distribs.Product(
                [distribs.Discrete('x', [TARGET_POSITIONS[i][0]]),
                distribs.Discrete('y', [TARGET_POSITIONS[i][1]])],
                shape='circle', scale=0.05, c0=color_list[i][0], c1=color_list[i][1], c2=color_list[i][2],
            )
            agents_generator = sprite_generators.generate_sprites(
                agents_factors, num_sprites=1)

            target_generator= sprite_generators.generate_sprites(
                target_factors, num_sprites=1) 

            if targets_overlap is None:
                target = target_generator(without_overlapping= walls  )
                agent = agents_generator(without_overlapping=  walls )
                targets_overlap = target
                agents_overlap = agent
            else:
                target = target_generator(
                    disjoint=True, without_overlapping= walls +  targets_overlap + agents_overlap)
                agent = agents_generator(without_overlapping=  walls +  targets_overlap + agents_overlap)
                targets_overlap = targets_overlap + target
                agents_overlap = agents_overlap + agent

            targets.append(target)#+ sum(targets) + sum(agents) ))
            agents.append(agent)

        state_list = [
            ('walls', walls),
        ]

        for i in range(num_sprites):
            state_list.append( ("target"+str(i), targets[i]))
            state_list.append( ("agent"+str(i), agents[i]))

        state = collections.OrderedDict(state_list)

        return state

    ############################################################################
    # Physics
    ############################################################################

    #agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    asymmetric_collision = physics_lib.Collision(
        elasticity=1., symmetric=False, update_angle_vel=True)
    symmetric_collision = physics_lib.Collision(
        elasticity=1., symmetric=True, update_angle_vel=True)
    agent_wall_collision = physics_lib.Collision(
        elasticity=0., symmetric=False, update_angle_vel=False)
    
    forces = []
    for i in range(num_sprites):
        forces.append(((agent_wall_collision, 'agent'+str(i), 'walls')))
        for j in range(num_sprites-1,0,-1):
            forces.append( (symmetric_collision, 'agent'+str(i), 'agent'+str(j)) )
    
    forces = tuple(forces)
    
    physics = physics_lib.Physics(*forces, updates_per_env_step=10)

    ############################################################################
    # Task
    ############################################################################
    contact_tasks = []
    if sparse_rewards:
        for i in range(num_sprites):
            contact_tasks.append(tasks.OneContactReward(1/num_sprites,
            layers_0='agent'+str(i), layers_1='target'+str(i),reset_steps_after_contact =0))

    else:
        for i in range(num_sprites):
            contact_tasks.append(tasks.FindGoal(
            layers_0='agent'+str(i), layers_1='target'+str(i),reset_steps_after_contact =0,raw_reward_multiplier = 5,terminate_distance=TERMINATE_DISTANCE))

    task = tasks.CompositeTask(*contact_tasks, timeout_steps=timeout_steps)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.SelectMove(
         action_layers=tuple(['agent'+str(i) for i in range(num_sprites)]) ,scale=0.05)

    ############################################################################
    # Observer
    ############################################################################

    observer_info = observers.SpriteInfo(sprite_layers=tuple(['agent'+str(i) for i in range(num_sprites)]))

    observer_dict = {"sprite_info":observer_info}
    if is_demo:
        observer_image = observers.PILRenderer(
            image_size=(64, 64), anti_aliasing=1, color_to_rgb=None,bg_color=(255,255,255))
        observer_dict['image'] = observer_image

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': observer_dict,
    }
    return config
