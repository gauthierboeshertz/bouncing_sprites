
from . import abstract_action_space
import numpy as np
from gym.spaces import MultiDiscrete
from enum   import IntEnum

class Act(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NO_ACTION = 4

class DiscreteMoveAllSprites(abstract_action_space.AbstractActionSpace):
  """
  Move sprites LEFT RIGHT UP DOWN 
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
    self._action_spec = MultiDiscrete([4 for _ in range(len(action_layers))])

    if not isinstance(action_layers, (list, tuple)):
        action_layers = (action_layers,)
    self._action_layers = action_layers


  def _get_motion(self, action,sprite):
    #delta_pos = (action[2:] - 0.5) * self._scale

      delta_pos = sprite.position - sprite.position
      return delta_pos

  def move_discrete(self, action, sprite):

      #print(action)
      #print(Act.NO_ACTION)
      
      if action == Act.LEFT:
          act = np.array([-1,0])
      elif action == Act.RIGHT:
          act = np.array([1,0])
      elif action == Act.UP:
          act = np.array([0,1])
      elif action == Act.DOWN:
          act = np.array([0,-1])      
      elif action == Act.NO_ACTION:
          #print("No action")
          return
      else:
          raise ValueError("Invalid action")
      sprite.velocity = (act / sprite.mass)*self._scale
      
      
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
        #print(action)
    #noised_action = self.apply_noise_to_action(action)
    for agent_idx, agent_layer in enumerate(self._action_layers):
        for sprite in state[agent_layer]:
            s_act = action[agent_idx]
            
            self.move_discrete(s_act, sprite)
            
  def reset(self, state):
      """Reset action space at start of new episode."""

  def random_action(self):
      """Return randomly sampled action."""
      return np.random.uniform(0., 1., size=self._action_spec.shape)
  
  def action_spec(self):
      return self._action_spec
