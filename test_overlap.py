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


RAW_REWARD_MULTIPLIER = 5
TERMINATE_DISTANCE = 0.05
color_list = [ (255*np.array(mcolors.to_rgb(color))).astype(np.int32).tolist() for name, color in mcolors.TABLEAU_COLORS.items()]#[[0,0,255],[255,0,0],[255,0,255],[0,0,0],[128,128,128],[255,0,255]]
TARGET_POSITIONS = []
#TARGET_POSITIONS = [[0.25,0.10],[0.75,0.10],[0.10,0.75],[0.50,0.75],[0.5,0.1],[0.8,0.5]]
SPRITE_RADIUS = 0.045300698895537665
TARGET_RADIUS = 0.03397552417165325

def gen_target_positions(delta_x,delta_y):
    positions = []
    for x in np.arange(0.90,0.1,-delta_x):
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
random.Random(0).shuffle(TARGET_POSITIONS)
TARGET_POSITIONS = np.array(TARGET_POSITIONS)

SPRITES_POSITIONS = gen_sprites_positions(0.2,0.2)
random.Random(2).shuffle(SPRITES_POSITIONS)
SPRITES_POSITIONS = np.array(SPRITES_POSITIONS)
#SPRITES_POSITIONS = [[0.8,0.5],[0.4,0.8],[0.5,0.5],[0.2,0.2],[0.8,0.2],[0.8,0.5]]
walls = shapes.border_walls(visible_thickness=0.05, c0=0, c1=0, c2=0)

print(TARGET_POSITIONS)
print("---")
print(SPRITES_POSITIONS)

target = [ sprite.Sprite(x=pos[0],y=pos[1],scale=0.06,shape='circle') for pos in TARGET_POSITIONS ]

sprites = [ sprite.Sprite(x=pos[0],y=pos[1],scale=0.08,shape='circle') for pos in SPRITES_POSITIONS ]

for w in walls:
    for s_i,s in enumerate(sprites):
        if w.overlaps_sprite(s):
            print(w, "wall overlaps", s_i , SPRITES_POSITIONS[s_i])

for t_i,t in enumerate(target):
    for s_i,s in enumerate(sprites):
        if t.overlaps_sprite(s):
            print(t_i, SPRITES_POSITIONS[t_i] , "overlaps", s_i , SPRITES_POSITIONS[s_i])

for t_i,t in enumerate(sprites):
    for s_i,s in enumerate(sprites):
        if t.overlaps_sprite(s) and t is not s:
            print(t_i, SPRITES_POSITIONS[t_i] , "overlaps", s_i , SPRITES_POSITIONS[s_i])

#TARGET_POSITIONS = np.array([[0.4999999999999999, 0.4999999999999999], [0.2999999999999998, 0.4999999999999999], [0.7, 0.7], [0.9, 0.7], [0.4999999999999999, 0.7], [0.9, 0.4999999999999999], [0.9, 0.2999999999999998], [0.4999999999999999, 0.2999999999999998], [0.2999999999999998, 0.7], [0.7, 0.2999999999999998], [0.4999999999999999, 0.9], [0.7, 0.9], [0.9, 0.9], [0.7, 0.4999999999999999], [0.2999999999999998, 0.2999999999999998], [0.2999999999999998, 0.9]])
#SPRITES_POSITIONS = np.array([[0.8, 0.2], [0.6000000000000001, 0.8], [0.4, 0.8], [0.8, 0.4], [0.4, 0.6000000000000001], [0.2, 0.2], [0.6000000000000001, 0.2], [0.2, 0.8], [0.6000000000000001, 0.4], [0.4, 0.2], [0.6000000000000001, 0.6000000000000001], [0.2, 0.6000000000000001], [0.4, 0.4], [0.8, 0.6000000000000001], [0.8, 0.8], [0.2, 0.4]])
