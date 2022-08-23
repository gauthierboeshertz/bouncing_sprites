# This file was forked and modified from the file here:
# https://github.com/deepmind/spriteworld/blob/master/spriteworld/renderers/pil_renderer.py
# Here is the license header for that file:

# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Python Image Library (PIL/Pillow) renderer."""

from . import abstract_observer
from . import color_maps
from . import polygon_modifiers

from dm_env import specs
import numpy as np
from PIL import Image
from PIL import ImageDraw


class SpriteInfo(abstract_observer.AbstractObserver):
    """
    Gives state as described in the paper Counterfactual Data Augmentation using Locally Factored Dynamics
    
    """

    def __init__(self,
                 sprite_layers= [],
                  ):
        """Construct PIL renderer.

        Args:
            image_size: Int tuple (height, width). Size of output of .render().
            anti_aliasing: Int. Anti-aliasing factor. Linearly scales the size
                of the internal canvas.
            bg_color: None or 3-tuple of ints in [0, 255]. Background color. If
                None, background is (0, 0, 0).
            color_to_rgb: String or Callable converting a tuple (c1, c2, c3) to
                a uint8 tuple (r, g, b) in [0, 255]. If string, must be the name
                of a function in color_maps.py, which will be looked up and
                used.
            polygon_modifier: Instance of
                polygon_modifiers.AbstractPolygonModifier. Callable taking state
                and returning a function converting list of polygons (sprite
                vertex arrays) to another list of polygon vertices. This can be
                used to adjust polygon positions to render first-person,
                duplicate sprites when simulating torus geometry, etc. See
                .polygon_modifiers.py for examples.
        """
        self._sprite_layers = sprite_layers
        self._observation_spec = specs.Array(
            shape=  (len(self._sprite_layers)*4,), dtype=np.uint8)

    def __call__(self, state):
        """
        returns Numpy array with pos and velocity of sprites
        Args:
            state: OrderedDict of iterables of sprites.

        Returns:
            Numpy array with pos and velocity of sprites
        """
        obs = np.zeros((len(self._sprite_layers),4))
        for sprite_layer_idx, sprite_layer in enumerate(self._sprite_layers):
            for sprite in state[sprite_layer]:
                obs[sprite_layer_idx] = [sprite.x,sprite.y, sprite.x_vel,sprite.y_vel]

        return obs.flatten()



    def observation_spec(self):
        return self._observation_spec
