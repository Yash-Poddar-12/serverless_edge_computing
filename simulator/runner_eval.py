import argparse
import numpy as np
from .runner_train import SimCore
from .autoscaler_q import QAutoscaler
from .plot_results import plot_metrics

def eval_qlearn(q_table_path='q_table.npy'):
    # load Q-table
    agent = QAutoscaler()
    agent.load(q_table_path)
    sim = SimCore()
    sim.reset_episode()
    # run a single episode using greedy policy
    while sim.now < 3600:  # 1 hour
        state_idx = len(sim.instances)
        action = int(np.argmax(agent.Q[state_idx]))
        # scale up
        for _ in range(action):
            if len(sim.instances) < agent.max_instances:
                sim.instances.append(__import__('simulator.instance', fromlist=['']).instance.FunctionInstance(sim.now, sim.warm_time))
        sim.step_with_action(0)
    summary = sim.metrics.summary()
    print("Evaluation summary:", summary)
    plot_metrics(sim.metrics, out='eval_qlearn.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['qlearn','dqn'], default='qlearn')
    parser.add_argument('--load', type=str, default='q_table.npy')
    args = parser.parse_args()

    if args.mode == 'qlearn':
        eval_qlearn(args.load)
    else:
        print("DQN eval not implemented here.")
