"""
Train Q-learning autoscaler (baseline) with adaptive warm-time support.
"""

import argparse
import numpy as np
from tqdm import trange

from .config import SIM_SECONDS_PER_EPISODE, TIME_STEP, DEFAULT_WARM_TIME, MAX_INSTANCES, ARRIVAL_LAMBDA
from .workload import WorkloadGenerator
from .instance import FunctionInstance
from .scheduler import Scheduler
from .autoscaler_q import QAutoscaler
from .metrics import MetricsCollector
from .utils import sample_service_time, poisson_arrivals


class SimCore:
    def __init__(self, lmbda=ARRIVAL_LAMBDA, warm_time=DEFAULT_WARM_TIME, scheduler_policy='warmest', adaptive=False):
        self.wg = WorkloadGenerator(lmbda)
        self.now = 0.0
        self.warm_time = warm_time
        self.instances = []
        self.scheduler = Scheduler(policy=scheduler_policy)
        self.metrics = MetricsCollector()
        self.queue = []
        self.adaptive = adaptive
        # EWMA for arrival estimation
        self.arrival_ewma = 0.0
        self.ewma_alpha = 0.2

    def reset_episode(self):
        self.now = 0.0
        self.instances = []
        self.metrics = MetricsCollector()
        self.queue = []
        self.arrival_ewma = 0.0
        return [0, 0.0, 0.0, 0]

    def _compute_adaptive_warm(self):
        # Heuristic: make warm_time proportional to estimated arrival rate
        est = max(0.0, self.arrival_ewma)
        factor = 120.0
        warm = int(max(30, min(900, factor * est)))
        return warm

    def step_with_action(self, action_scale_up):
        # apply incoming scale-up action
        for _ in range(action_scale_up):
            if len(self.instances) < MAX_INSTANCES:
                wt = self.warm_time
                if self.adaptive:
                    wt = self._compute_adaptive_warm()
                self.instances.append(FunctionInstance(self.now, wt))

        # arrivals using Poisson with configured lambda
        n = poisson_arrivals(self.wg.lmbda)
        # update arrival EWMA
        self.arrival_ewma = self.ewma_alpha * n + (1 - self.ewma_alpha) * self.arrival_ewma

        reqs = n
        succ = 0
        fail = 0
        cold = 0
        latencies = []

        durations = [sample_service_time() for _ in range(n)]
        for d in durations:
            inst = self.scheduler.pick_instance(self.instances, self.now)
            if inst is None:
                if len(self.instances) < MAX_INSTANCES:
                    wt = self.warm_time
                    if self.adaptive:
                        wt = self._compute_adaptive_warm()
                    new_inst = FunctionInstance(self.now, wt)
                    self.instances.append(new_inst)
                    inst = new_inst
                else:
                    fail += 1
                    continue
            was_warm = inst.is_warm(self.now)
            if not was_warm:
                cold += 1
            start = inst.assign(self.now, d)
            finish = start + d
            latency = finish - self.now  # simple service latency approximation
            latencies.append(latency)
            succ += 1

        busy = sum(1 for i in self.instances if i.busy_until > self.now)
        util = busy / max(1, len(self.instances)) if len(self.instances) > 0 else 0.0
        idle_instances = max(0, len(self.instances) - busy)

        reward = QAutoscaler.compute_reward(succ, fail, util, idle_instances)

        self.now += TIME_STEP

        # remove expired instances (scale-down via warm-time expiry)
        new_instances = []
        for inst in self.instances:
            if inst.is_warm(self.now) or inst.busy_until > self.now:
                new_instances.append(inst)
        self.instances = new_instances

        state = [len(self.instances), reqs, util, len(self.queue)]
        done = (self.now >= SIM_SECONDS_PER_EPISODE)
        info = {}
        self.metrics.add_tick(reqs, succ, fail, cold, len(self.instances), util, latencies)
        return state, reward, done, info


def train_q_learning(episodes=200, scheduler_policy='warmest', adaptive=False):
    sim = SimCore(adaptive=adaptive)
    agent = QAutoscaler()
    for ep in trange(episodes, desc='Episodes'):
        sim.reset_episode()
        state = len(sim.instances)
        while sim.now < SIM_SECONDS_PER_EPISODE:
            action = agent.choose_action(state)
            for _ in range(action):
                if len(sim.instances) < MAX_INSTANCES:
                    wt = sim.warm_time
                    if sim.adaptive:
                        wt = sim._compute_adaptive_warm()
                    sim.instances.append(FunctionInstance(sim.now, wt))
            next_state, reward, done, info = sim.step_with_action(0)
            next_state_idx = next_state[0]
            agent.update(state, action, reward, next_state_idx)
            state = next_state_idx
        agent.decay_epsilon()
    agent.save('q_table.npy')
    print("Training complete. Q-table saved to q_table.npy")
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['qlearn', 'dqn'], default='qlearn')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--adaptive', action='store_true', help='Enable adaptive warm-time')
    args = parser.parse_args()

    if args.mode == 'qlearn':
        train_q_learning(episodes=args.episodes, adaptive=args.adaptive)
    else:
        print("Use dqn_train.py for DQN training flow.")
