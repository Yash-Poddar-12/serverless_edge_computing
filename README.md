# Serverless Edge Simulator + Testbed Controller

This repo contains:

- a Python discrete-event/time-stepped simulator reproducing the Q-learning autoscaler + warmest scheduler baseline;
- adaptive warm-time extension and optional DQN autoscaler (uses stable-baselines3);
- a minimal OpenFaaS / k3s controller example for a small real testbed.

## Quick start (simulator)

1. Create venv & install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
