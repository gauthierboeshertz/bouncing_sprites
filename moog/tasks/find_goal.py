"""Task for receiving rewards upon contact."""

from . import abstract_task
import inspect
import numpy as np

class FindGoalPosition(AbstractTask):
  """Used for tasks that require moving some sprites to a goal position."""

  def __init__(self,
               filter_distrib=None,
               goal_position=(0.5, 0.5),
               terminate_distance=0.05,
               terminate_bonus=0.0,
               weights_dimensions=(1, 1),
               sparse_reward=False,
               raw_reward_multiplier=50):
    """Construct goal-finding task.
    This task rewards the agent for bringing all sprites with factors contained
    in a filter distribution to a goal position. Rewards are offset to be
    negative, except for a termination bonus when the goal is reached.
    Args:
      filter_distrib: None or instance of
        factor_distributions.AbstractDistribution. If None, all sprites must be
        brought to the goal position. If not None, only sprites with factors
        contained in this distribution must be brought to the goal position.
      goal_position: Position of the goal.
      terminate_distance: Distance from goal position at which to clip reward.
        If all sprites are within this distance, terminate episode.
      terminate_bonus: Extra bonus for getting all sprites within
        terminate_distance.
      weights_dimensions: Weights modifying the contributions of the (x,
        y)-dimensions to the distance to goal computation.
      sparse_reward: Boolean (default False), whether to provide dense rewards
        or only reward at the end of an episode.
      raw_reward_multiplier: Multiplier for the reward to be applied before
        terminate_bonus. Empirically, 50 seems to be a good value.
    """
    self._filter_distrib = filter_distrib
    self._goal_position = np.asarray(goal_position)
    self._terminate_bonus = terminate_bonus
    self._terminate_distance = terminate_distance
    self._sparse_reward = sparse_reward
    self._weights_dimensions = np.asarray(weights_dimensions)
    self._raw_reward_multiplier = raw_reward_multiplier

  def _single_sprite_reward(self, sprite):
    goal_distance = np.sum(self._weights_dimensions *
                           (sprite.position - self._goal_position)**2.)**0.5
    raw_reward = self._terminate_distance - goal_distance
    return self._raw_reward_multiplier * raw_reward

  def _filtered_sprites_rewards(self, sprites):
    """Returns list of rewards for the filtered sprites."""
    rewards = [
        self._single_sprite_reward(s) for s in sprites if
        self._filter_distrib is None or self._filter_distrib.contains(s.factors)
    ]
    return rewards

  def reward(self, sprites):
    """Calculate total reward summed over filtered sprites."""
    reward = 0.

    rewards = self._filtered_sprites_rewards(sprites)
    if not rewards:  # No sprites get through the filter, so make reward NaN
      return np.nan
    dense_reward = np.sum(rewards)

    if all(np.array(rewards) >= 0):  # task succeeded
      reward += self._terminate_bonus
      reward += dense_reward
    elif not self._sparse_reward:
      reward += dense_reward

    return reward

  def success(self, sprites):
    return all(np.array(self._filtered_sprites_rewards(sprites)) >= 0)
