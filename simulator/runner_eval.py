import argparse
from .runner_train import SimCore
from .autoscaler_q import QAutoscaler
from .plot_results import plot_metrics
from .instance import FunctionInstance

def eval_qlearn(q_table_path='q_table.npy', adaptive=False):
    agent = QAutoscaler()
    agent.load(q_table_path)
    sim = SimCore(adaptive=adaptive)
    sim.reset_episode()
    # run a single episode (3600 sec)
    while sim.now < 3600:
        state_idx = len(sim.instances)
        action = int(agent.Q[state_idx].argmax())
        for _ in range(action):
            if len(sim.instances) < agent.max_instances:
                wt = sim.warm_time
                if sim.adaptive:
                    wt = sim._compute_adaptive_warm()
                sim.instances.append(FunctionInstance(sim.now, wt))
        sim.step_with_action(0)
    summary = sim.metrics.summary()
    print("Evaluation summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    plot_metrics(sim.metrics, out='eval_qlearn.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='q_table.npy')
    parser.add_argument('--adaptive', action='store_true')
    args = parser.parse_args()
    eval_qlearn(args.load, adaptive=args.adaptive)
