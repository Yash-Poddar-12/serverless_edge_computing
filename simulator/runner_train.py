"""
Train Q-learning autoscaler (baseline) or DQN (optional).
This runner implements a simple time-stepped simulation.
"""

import argparse
import random
import numpy as np
from tqdm import trange

from .config import SIM_SECONDS_PER_EPISODE, TIME_STEP, DEFAULT_WARM_TIME, MAX_INSTANCES
from .workload import WorkloadGenerator
from .instance import FunctionInstance
from .scheduler import Scheduler
from .autoscaler_q import QAutoscaler
from .metrics import MetricsCollector
from .utils import sample_service_time
from .utils import poisson_arrivals

# Simple simulation core class
class SimCore:
    def __init__(self, lmbda=0.5, warm_time=DEFAULT_WARM_TIME, scheduler_policy='warmest'):
        self.wg = WorkloadGenerator(lmbda)
        self.now = 0.0
        self.warm_time = warm_time
        self.instances = []  # list of FunctionInstance
        self.scheduler = Scheduler(policy=scheduler_policy)
        self.metrics = MetricsCollector()
        self.queue = []  # queued job durations

    def reset_episode(self):
        self.now = 0.0
        self.instances = []
        self.metrics = MetricsCollector()
        self.queue = []
        # return initial state (for DQN wrapper), e.g., zeros
        return [0, 0.0, 0.0, 0]

    def step_with_action(self, action_scale_up):
        """
        Execute one time step with an external scale-up action and return state,reward,done,info
        For DQN gym env compatibility.
        """
        # scale up
        for _ in range(action_scale_up):
            if len(self.instances) < MAX_INSTANCES:
                self.instances.append(FunctionInstance(self.now, self.warm_time))

        # arrivals
        n = poisson_arrivals(0.5)
        reqs = n
        succ = 0
        fail = 0
        cold = 0

        # assign incoming requests
        durations = [sample_service_time() for _ in range(n)]
        for d in durations:
            inst = self.scheduler.pick_instance(self.instances, self.now)
            if inst is None or inst.busy_until > self.now + 1e8:
                # if no instance available, spawn one immediately (scale action didn't allow)
                if len(self.instances) < MAX_INSTANCES:
                    new_inst = FunctionInstance(self.now, self.warm_time)
                    self.instances.append(new_inst)
                    inst = new_inst
            if inst:
                was_warm = inst.is_warm(self.now)
                if not was_warm:
                    cold += 1
                inst.assign(self.now, d)
                succ += 1
            else:
                # queueing / failing (simplified: fail if no instance)
                fail += 1

        # simple utilization: fraction of busy instances at this tick
        busy = sum(1 for i in self.instances if i.busy_until > self.now)
        util = busy / max(1, len(self.instances))

        idle_instances = max(0, len(self.instances) - busy)
        # reward approximated (successful, failed, util, idle)
        reward = QAutoscaler.compute_reward(succ, fail, util, idle_instances)

        # step time
        self.now += TIME_STEP

        # remove expired instances (scale-down via warm-time expiry)
        new_instances = []
        for inst in self.instances:
            if inst.is_warm(self.now) or inst.busy_until > self.now:
                new_instances.append(inst)
        self.instances = new_instances

        # state = [active_instances, recent_arrival_rate(~reqs), avg_util, queue_len]
        state = [len(self.instances), reqs, util, 0]
        done = (self.now >= SIM_SECONDS_PER_EPISODE)
        info = {}
        # metrics
        self.metrics.add_tick(reqs, succ, fail, cold, len(self.instances), util)
        return state, reward, done, info

def train_q_learning(episodes=200, scheduler_policy='warmest'):
    sim = SimCore()
    agent = QAutoscaler()
    for ep in trange(episodes, desc='Episodes'):
        sim.reset_episode()
        state = len(sim.instances)
        step = 0
        while sim.now < SIM_SECONDS_PER_EPISODE:
            action = agent.choose_action(state)
            # scale-up
            for _ in range(action):
                if len(sim.instances) < MAX_INSTANCES:
                    sim.instances.append(FunctionInstance(sim.now, sim.warm_time))
            # simulate one tick with no external DQN input
            # call the step_with_action with action=0 because we've already scaled
            next_state, reward, done, info = sim.step_with_action(0)
            next_state_idx = next_state[0]
            agent.update(state, action, reward, next_state_idx)
            state = next_state_idx
            step += 1
        agent.decay_epsilon()
    agent.save('q_table.npy')
    print("Training complete. Q-table saved to q_table.npy")
    return agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['qlearn', 'dqn'], default='qlearn')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--timesteps', type=int, default=200000)
    args = parser.parse_args()

    if args.mode == 'qlearn':
        train_q_learning(episodes=args.episodes)
    else:
        print("DQN training not implemented in this runner. Use DQN flow in a separate script.")
