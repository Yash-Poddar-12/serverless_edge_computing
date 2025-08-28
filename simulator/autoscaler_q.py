import numpy as np
from .config import Q_ALPHA, Q_GAMMA, Q_EPSILON_START, Q_EPSILON_MIN, Q_EPSILON_DECAY, MAX_INSTANCES
from .config import W1, W2, W3, W4

class QAutoscaler:
    def __init__(self, max_instances=MAX_INSTANCES, alpha=Q_ALPHA, gamma=Q_GAMMA):
        self.max_instances = max_instances
        self.alpha = alpha
        self.gamma = gamma
        # state = number of active instances (0..max_instances)
        # actions = scale up by 0..k (we choose k=5 for granularity)
        self.action_k = 5
        self.Q = np.zeros((self.max_instances + 1, self.action_k + 1))
        self.epsilon = Q_EPSILON_START

    def choose_action(self, state):
        import random
        if random.random() < self.epsilon:
            return random.randint(0, self.action_k)
        return int(np.argmax(self.Q[state, :]))

    def update(self, state, action, reward, next_state):
        old = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] = old + self.alpha * (target - old)

    def decay_epsilon(self):
        self.epsilon = max(Q_EPSILON_MIN, self.epsilon * Q_EPSILON_DECAY)

    def save(self, path_q='q_table.npy'):
        np.save(path_q, self.Q)

    def load(self, path_q='q_table.npy'):
        self.Q = np.load(path_q)

    @staticmethod
    def compute_reward(successful, failed, util_ratio, idle_instances):
        # reward as linear combination (paper-like)
        return W1 * successful - W2 * failed + W3 * util_ratio - W4 * idle_instances
