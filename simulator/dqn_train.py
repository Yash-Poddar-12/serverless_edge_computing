"""
Train a DQN agent using stable-baselines3 on the SimpleServerlessEnv wrapper.
Requires: stable-baselines3 and torch installed.
"""
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from .runner_train import SimCore
from .dqn_env import SimpleServerlessEnv

def train_dqn(timesteps=200_000, model_path='dqn_model.zip'):
    sim = SimCore()
    env = SimpleServerlessEnv(sim)
    model = DQN('MlpPolicy', env, verbose=1)
    cb = CheckpointCallback(save_freq=50_000, save_path='checkpoints/', name_prefix='dqn')
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(model_path)
    print('Saved DQN to', model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200000)
    args = parser.parse_args()
    train_dqn(timesteps=args.timesteps)
