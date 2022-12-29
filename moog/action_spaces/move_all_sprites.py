
from . import abstract_action_space
from dm_env import specs
import numpy as np


class MoveAllSprites(abstract_action_space.AbstractActionSpace):
  """
  Moves the first sprite in the list that has not reached his target
  action shape is (2* number of sprites,) which is the direction to give the sprite
  """

  def __init__(self, action_layers=[],
                scale=1.0, motion_cost=0.0, noise_scale=None,instant_move=False):
    """Constructor.

    Args:
      scale: Multiplier by which the motion is scaled down. Should be in [0.0,
        1.0].
      motion_cost: Factor by which motion incurs cost.
      noise_scale: Optional stddev of the noise. If scalar, applied to all
        action space components. If vector, must have same shape as action.
    """
    self._scale = scale
    self._motion_cost = motion_cost
    self._noise_scale = noise_scale
    self._instant_move = instant_move
    self._action_spec = specs.BoundedArray(
        shape=(2*len(action_layers),), dtype=np.float32, minimum=-1.0, maximum=1.0)

    if not isinstance(action_layers, (list, tuple)):
        action_layers = (action_layers,)
    self._action_layers = action_layers



  def apply_noise_to_action(self, action):
    if self._noise_scale:
      noise = np.random.normal(
          loc=0.0, scale=self._noise_scale, size=action.shape)
      return action + noise
    else:
      return action

  def get_motion(self, action,sprite):
    #delta_pos = (action[2:] - 0.5) * self._scale
    delta_pos = action - sprite.position
    return delta_pos

  def get_sprite_from_position(self, position, sprites):
    for sprite in sprites[::-1]:
      if sprite.contains_point(position):
        return sprite
    return None

  def step(self, state, action):
    """Take an action and move the sprites.
    Args:
      action: Numpy array of shape (4,) in [0, 1]. First two components are the
        position selection, second two are the motion selection.
      sprites: Iterable of sprite.Sprite() instances. If a sprite is moved by
        the action, its position is updated.
      keep_in_frame: Bool. Whether to force sprites to stay in the frame by
        clipping their centers of mass to be in [0, 1].

    Returns:
      Scalar cost of taking this action.
    """
    if action[0] == -10000:
        return
        #print(action)
    #noised_action = self.apply_noise_to_action(action)
    noised_action = (action)
    for agent_idx, agent_layer in enumerate(self._action_layers):
        for sprite in state[agent_layer]:
            motion = self.get_motion(noised_action[2*agent_idx:2*(agent_idx+1)],sprite)
            if not self._instant_move:
              sprite.velocity += (motion / sprite.mass)*self._scale #self._action / sprite.mass
            else:
              sprite.velocity = (motion / sprite.mass)*self._scale

  def reset(self, state):
      """Reset action space at start of new episode."""

  def random_action(self):
      """Return randomly sampled action."""
      return np.random.uniform(0., 1., size=self._action_spec.shape)
  
  def action_spec(self):
      return self._action_spec
