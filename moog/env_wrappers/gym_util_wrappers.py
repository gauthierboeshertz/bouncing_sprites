import numpy as np
import gym
from gym.wrappers.frame_stack import LazyFrames
from gym.spaces import Box

class FrameStackActionRepeat(gym.Wrapper):
    
    def __init__(self, env, num_frame_stack=4,num_action_repeat=1):
        
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self.num_stack = num_frame_stack
        self.num_action_repeat = num_action_repeat
        
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_frame_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_frame_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def step(self, action):
        """Repeat action, sum reward, and stack observations.
        For now num repeat less than num stack obs is not implemente
        
        """
        total_reward = 0.0
        done = None
        obs_stack = []
        for i in range(self.num_stack):
            obs, reward, done, info = self.env.step(action if i < self.num_action_repeat else None)
            total_reward += reward

            if done:
                break
            obs_stack.append(obs)
        # Note that the observation on the done=True frame
        # doesn't matter
        obs_stack = np.array(obs_stack)

        return obs_stack, total_reward, done, info

    def reset(self):
        first_ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(first_ob)
        return self._get_ob()
