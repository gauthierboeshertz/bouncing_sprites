"""Composite task."""

from . import abstract_task
import numpy as np


class CompositeTask(abstract_task.AbstractTask):
    """CompositeTask task.

    This combines multiple tasks at once, summing the rewards from each of them.
    This can be useful for example to have a predator/prey task where there are
    positive rewards for catching the prey and negative rewards for being caught
    by the predators.
    """

    def __init__(self, *tasks, timeout_steps=np.inf,all_reset=True,divide_by_tasks=False):
        """Constructor.

        Args:
            tasks: Tasks to compose. Reward will be the sum of the rewards from
                each of these tasks.
            timeout_steps: After this number of steps since reset, a reset is
                forced.
        """

        self._tasks = tasks
        self.divide_by_tasks = divide_by_tasks
        self._timeout_steps = timeout_steps
        self._reseted_tasks = [False for _ in self._tasks]

    def reset(self, state, meta_state):
        for task in self._tasks:
            task.reset(state, meta_state)
        self._reseted_tasks = [False for _ in self._tasks]

    def reward(self, state, meta_state, step_count):
        """Compute reward."""
        reward = 0
        timed_out = step_count >= self._timeout_steps
        for task_idx, task in enumerate(self._tasks):
            task_reward, task_should_reset = task.reward(
                state, meta_state, step_count)
            reward += task_reward
            self._reseted_tasks[task_idx] = task_should_reset  or  self._reseted_tasks[task_idx]
        
        if self.divide_by_tasks:
            reward = reward/len(self._tasks)
            
        return reward, all(self._reseted_tasks) or timed_out
