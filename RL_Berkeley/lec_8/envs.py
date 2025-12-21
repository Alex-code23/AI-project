import math
import random
import numpy as np

class GridWorld:
    """
    Simple NxN GridWorld.
    State: one-hot vector of size N*N (option one_hot=True) or normalized coords.
    Actions: 0=up,1=right,2=down,3=left
    Reward: +1 at goal (bottom-right), else -0.01 per step.
    """
    def __init__(self, size=5, max_steps=100, one_hot=True):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.max_steps = max_steps
        self.one_hot = one_hot
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        r, c = self.pos
        if action == 0 and r > 0:
            r -= 1
        elif action == 1 and c < self.size - 1:
            c += 1
        elif action == 2 and r < self.size - 1:
            r += 1
        elif action == 3 and c > 0:
            c -= 1
        self.pos = (r, c)
        self.steps += 1
        done = (self.pos == (self.size - 1, self.size - 1)) or (self.steps >= self.max_steps)
        reward = 1.0 if self.pos == (self.size - 1, self.size - 1) else -0.01
        return self._get_obs(), reward, done, {}

    def sample_action(self):
        return random.randrange(self.n_actions)

    def _get_obs(self):
        idx = self.pos[0] * self.size + self.pos[1]
        if self.one_hot:
            v = np.zeros(self.n_states, dtype=np.float32)
            v[idx] = 1.0
            return v
        else:
            return np.array([self.pos[0] / (self.size - 1), self.pos[1] / (self.size - 1)], dtype=np.float32)

class PointMass1D:
    """
    Simple 1D point mass with continuous actions.
    State: [x, v], x in [-5,5], v clipped.
    Action: continuous force in [-1,1]
    Dynamics: v += a*dt, x += v*dt (with mild damping)
    Reward: negative absolute distance to target (0) minus small control penalty
    """
    def __init__(self, dt=0.1, max_steps=200, target=0.0):
        self.dt = dt
        self.max_steps = max_steps
        self.target = target
        self.action_space = (-1.0, 1.0)
        self.observation_space = (-5.0, 5.0)
        self.reset()

    def reset(self):
        self.x = random.uniform(-4.0, 4.0)
        self.v = 0.0
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        # clip action
        a = max(min(action, self.action_space[1]), self.action_space[0])
        self.v += a * self.dt
        # damping
        self.v *= 0.99
        self.x += self.v * self.dt
        self.x = max(min(self.x, self.observation_space[1]), self.observation_space[0])
        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = -abs(self.x - self.target) - 0.01 * (a ** 2)
        return self._get_obs(), reward, done, {}

    def sample_action(self):
        return random.uniform(self.action_space[0], self.action_space[1])

    def _get_obs(self):
        return np.array([self.x, self.v], dtype=np.float32)
