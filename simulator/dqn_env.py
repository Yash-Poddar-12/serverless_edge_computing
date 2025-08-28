# Minimal gym environment wrapper around the simulator state so SB3 DQN can be used.
import gym
from gym import spaces
import numpy as np

class SimpleServerlessEnv(gym.Env):
    """
    State: [active_instances, recent_arrival_rate, avg_utilization, queue_len]
    Action: Discrete(up to K) scale-up count
    """
    def __init__(self, sim_core, max_action=5):
        super().__init__()
        self.sim = sim_core  # a reference to the simulation core you implement in runner_train
        self.max_action = max_action
        self.observation_space = spaces.Box(low=0.0, high=1e6, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_action + 1)

    def reset(self):
        obs = self.sim.reset_episode()
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.sim.step_with_action(action)
        return np.array(obs, dtype=np.float32), reward, done, info
