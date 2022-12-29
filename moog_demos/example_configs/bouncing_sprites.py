"""Avoid colliding predator polygons.

This task serves to showcase collisions. The predators have a variety of
polygonal shapes and bounce off each other and off the walls with Newtonian
collisions. The subject controls a green agent circle. The subject gets negative
reward if contacted by a predators and positive reward periodically.
"""

import collections
import numpy as np
import matplotlib.colors as mcolors
import random

from moog import action_spaces
from moog import physics as physics_lib
from moog import observers
from moog import sprite 
from moog import tasks
from moog import shapes
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators
import copy

RAW_REWARD_MULTIPLIER = 5
TERMINATE_DISTANCE = 0.05
#color_list = [ (255*np.array(mcolors.to_rgb(color))).astype(np.int32).tolist() for name, color in mcolors.TABLEAU_COLORS.items()]#[[0,0,255],[255,0,0],[255,0,255],[0,0,0],[128,128,128],[255,0,255]]
color_list = [ (255*np.array([1,0,0])).tolist(), (255*np.array([0,1,0])).tolist(), (255*np.array([0,0,1])).tolist(), (255*np.array([1,1,1])).tolist(), (255*np.array([0.5,0.5,0])).tolist(), (255*np.array([0.5,0,0.5])).tolist()]


TARGET_POSITIONS = []
#TARGET_POSITIONS = [[0.25,0.10],[0.75,0.10],[0.10,0.75],[0.50,0.75],[0.5,0.1],[0.8,0.5]]
SPRITE_RADIUS = 0.045300698895537665
TARGET_RADIUS = 0.03397552417165325

def gen_target_positions(delta_x,delta_y):
    positions = []
    for x in np.arange(0.8,0.15,-delta_x):
        for y in np.arange(0.90,0.1,-delta_y):
            positions.append([x,y])
    return positions

def gen_sprites_positions(delta_x,delta_y):
    positions = []
    for x in np.arange(0.2,0.91,delta_x):
        for y in np.arange(0.2,0.91,delta_y):
            positions.append([x,y])
    return positions

TARGET_POSITIONS = gen_target_positions(0.2,0.2)
random.Random(2).shuffle(TARGET_POSITIONS)
TARGET_POSITIONS = np.array(TARGET_POSITIONS)

SPRITES_POSITIONS = gen_sprites_positions(0.2,0.2)
random.Random(2).shuffle(SPRITES_POSITIONS)
#SPRITES_POSITIONS = np.array(SPRITES_POSITIONS)
#SPRITES_POSITIONS = [[0.8,0.5],[0.4,0.8],[0.5,0.5],[0.2,0.2],[0.8,0.2],[0.8,0.5]]
BACKGROUND_COLOR = (0, 0, 0)


def get_config(num_sprites,is_demo=True,timeout_steps=1000,sparse_reward=False,contact_reward=False,random_init_places=False,one_sprite_mover=False, all_sprite_mover=False, 
               discrete_all_sprite_mover=False, visual_obs=False,instant_move=False,action_scale=0.01,
               add_sprite_info=False,seed=0,dont_show_targets=False,disappear_after_contact=False):
    """Get environment config."""

    print("Using bouncing ball environment with {} sprites".format(num_sprites))
    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent
    random.seed(seed)
    np.random.seed(seed)
    # Walls
    walls = shapes.border_walls(visible_thickness=0.05, c0=0, c1=0, c2=0)

    
    # Create callable initializer returning entire state

    def state_initializer():
        targets = []
        agents = []
        targets_overlap = None
        agents_overlap = None

        ## First create targets so other sprites cant go on them
        sprite_shape = "circle"
        for i in range(num_sprites):
            target_color_list_i = BACKGROUND_COLOR if dont_show_targets else color_list[i] 
            target_factors = distribs.Product(
                [distribs.Discrete('x', [TARGET_POSITIONS[i,0]]),
                distribs.Discrete('y', [TARGET_POSITIONS[i,1]])],
                shape="square", scale=0.06, c0=target_color_list_i[0], c1=target_color_list_i[1], c2=target_color_list_i[2])
            

            target_generator= sprite_generators.generate_sprites(
                target_factors, num_sprites=1) 

            if targets_overlap is None:
                target = target_generator(without_overlapping= walls  )
                targets_overlap = target
            else:
                target = target_generator(
                    disjoint=True, without_overlapping= walls +  targets_overlap )
                targets_overlap = targets_overlap + target

            targets.append(target)#+ sum(targets) + sum(agents) ))

        sprite_positions = copy.deepcopy(SPRITES_POSITIONS)
        if random_init_places:
            random.shuffle(sprite_positions)

        for i in range(num_sprites):
            agents_factors = distribs.Product(
                    [distribs.Discrete('x', [sprite_positions[i][0]]),
                    distribs.Discrete('y', [sprite_positions[i][1]])],
                    #[distribs.Discrete('x', [target_positions[i][0]-0.1]),
                    #distribs.Discrete('y', [target_positions[i][1]-0.1])],

                    shape=sprite_shape, scale=0.08, c0=color_list[i][0], c1=color_list[i][1], c2=color_list[i][2],
                )

            agents_generator = sprite_generators.generate_sprites(
                agents_factors, num_sprites=1)

            if agents_overlap is None:
                agent = agents_generator(without_overlapping=  walls + targets_overlap)
                agents_overlap = agent
            else:
                agent = agents_generator(without_overlapping=  walls +  targets_overlap + agents_overlap)
                agents_overlap = agents_overlap + agent

            agents.append(agent)


        state_list = [('walls', walls)] + [('target{}'.format(i), targets[i]) for i in range(num_sprites)] + [('agent{}'.format(i), agents[i]) for i in range(num_sprites)]

        state = collections.OrderedDict(state_list)

        return state

    ############################################################################
    # Physics
    ############################################################################

    #agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    symmetric_collision = physics_lib.Collision(
        elasticity=1., symmetric=True, update_angle_vel=True)
    agent_wall_collision = physics_lib.Collision(
        elasticity=1., symmetric=False, update_angle_vel=True)
    
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
    if sparse_reward:
        for i in range(num_sprites):
            contact_tasks.append(tasks.OneContactReward(1/num_sprites,
            layers_0='agent'+str(i), layers_1='target'+str(i),reset_steps_after_contact =0))
        task = tasks.SparseContactReward(*contact_tasks, timeout_steps=timeout_steps)
    elif contact_reward:
        for i in range(num_sprites):
            contact_tasks.append(tasks.OneContactReward(1/num_sprites,
            layers_0='agent'+str(i), layers_1='target'+str(i),reset_steps_after_contact =0,disappear_after_contact=disappear_after_contact))
        task = tasks.CompositeTask(*contact_tasks, timeout_steps=timeout_steps)
    else:
        for i in range(num_sprites):
            contact_tasks.append(tasks.L2Reward(
            layers_0='agent'+str(i), layers_1='target'+str(i),reset_steps_after_contact =0,raw_reward_multiplier = RAW_REWARD_MULTIPLIER,terminate_distance=TERMINATE_DISTANCE))

        task = tasks.CompositeTask(*contact_tasks, timeout_steps=timeout_steps,divide_by_tasks=True)

    ############################################################################
    # Action space
    ############################################################################

    if  one_sprite_mover:
        action_space = action_spaces.MoveOneSprite(    
            action_layers=tuple(['agent'+str(i) for i in range(num_sprites)]),agent_tasks=contact_tasks ,scale=action_scale)
    elif all_sprite_mover:
        action_space = action_spaces.MoveAllSprites(    
            action_layers=tuple(['agent'+str(i) for i in range(num_sprites)]) ,instant_move=instant_move,scale=action_scale)
    elif discrete_all_sprite_mover:
        action_space = action_spaces.DiscreteMoveAllSprites(    
            action_layers=tuple(['agent'+str(i) for i in range(num_sprites)]) ,instant_move=instant_move,scale=action_scale)
    else:
        action_space = action_spaces.SelectMove(    
            action_layers=tuple(['agent'+str(i) for i in range(num_sprites)]) ,instant_move=instant_move,scale=action_scale)

    ############################################################################
    # Observer
    ############################################################################

    observer_dict = {}
    if visual_obs or is_demo:
        observer_image = observers.PILRenderer(
            image_size=(128, 128), anti_aliasing=1, color_to_rgb=None,bg_color=BACKGROUND_COLOR)
        observer_dict['image'] = observer_image
        
    if not visual_obs or add_sprite_info:
        observer_info = observers.SpriteInfo(sprite_layers=tuple(['agent'+str(i) for i in range(num_sprites)]))
        observer_dict["sprite_info"] = observer_info
        

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
