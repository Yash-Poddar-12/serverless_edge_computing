Serverless Edge Simulator + Testbed Controller
This repo contains:

- a Python discrete-event/time-stepped simulator reproducing the Q-learning autoscaler + warmest
  scheduler baseline;
- adaptive warm-time extension and optional DQN autoscaler (uses stable-baselines3);
- a minimal OpenFaaS / k3s controller example for a small real testbed.

## Quick start (simulator)

1. Create venv & install:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   Train Q-learning autoscaler (reproduced baseline):
   python simulator/runner_train.py --mode qlearn --episodes 200
   Evaluate & plot:
   python simulator/runner_eval.py --mode qlearn --load q_table.npy
   Optional: DQN
   Install extra deps (stable-baselines3, torch) and run:
   python simulator/runner_train.py --mode dqn --timesteps 200000
   python simulator/runner_eval.py --mode dqn --load dqn_model.zip
   OpenFaaS testbed (manual)
   Deploy k3s/minikube + OpenFaaS on your machine.
   Build and push the function in openfaas_testbed/function or use faas-cli.
   Use openfaas_testbed/controller/scale_controller.py to control replicas (example uses kubectl).
   Notes
   This code is intentionally minimal and well documented so you can reproduce baseline results quickly,
   then add your novelties (adaptive warm time, energy-aware reward, federated RL, etc).
   The default parameters mirror the baseline (see simulator/config.py). When reporting in your paper, cite
   the baseline as the reproduced system and state which modules you changed.

---

Simulator code
Create directory `simulator/` and place the following files.

## simulator/config.py

# Basic parameters (tweak for experiments)

SIM_SECONDS_PER_EPISODE = 60 \* 60 # 1 hour per episode (seconds)
TIME_STEP = 1 # simulation tick in seconds
MAX_INSTANCES = 50

# Workload mix (from the reference paper's style)

WORKLOAD_SHORT_RATIO = 0.90
WORKLOAD_MED_RATIO = 0.09
WORKLOAD_LONG_RATIO = 0.01

# Short tasks: 0.1 - 1 s -> we will scale by seconds

SHORT_MIN, SHORT_MAX = 0.1, 1.0
MED_MIN, MED_MAX = 1.0, 10.0
LONG_MIN, LONG_MAX = 10.0, 100.0

# Warm-time default (seconds)

DEFAULT_WARM_TIME = 180

# Q-learning params (baseline reproduction)

Q_ALPHA = 0.1
Q_GAMMA = 0.95
Q_EPSILON_START = 0.9
Q_EPSILON_MIN = 0.01
Q_EPSILON_DECAY = 0.995

# Reward weights (paper baseline)

W1, W2, W3, W4 = 0.4, 0.2, 0.2, 0.2

# Arrival rate baseline (lambda req/sec)

ARRIVAL_LAMBDA = 0.5 # average requests per second (tweak for scenarios)
