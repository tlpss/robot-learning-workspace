from ur_sim.push_env import UR3ePush

import pathlib
import stable_baselines3 as sb3
from stable_baselines3 import SAC
import wandb
import stable_baselines3.common.off_policy_algorithm
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
from wandb.integration.sb3 import WandbCallback
import torch


if __name__ == "__main__":

    folder_path = pathlib.Path(__file__).parent
    config = {
        "gamma": 0.9,
        "lr": 3e-3,
        "learning_starts": 300,
        "batch_size": 128,
        "time_steps": 6000,
        "seed": 2022,
        "use_state_observations": True,
        "use_push_primitive":True,

    }


    torch.manual_seed(config["seed"])

    env = UR3ePush(state_observation=config["use_state_observations"],push_primitive=config["use_push_primitive"],real_time=True)

    tb_path = folder_path / "ur_pusher_state_tb/"
    # https://docs.wandb.ai/guides/integrations/other/stable-baselines-3


    #run = wandb.init(project="ur_pusher", config=config, sync_tensorboard=True)

    model = SAC("MlpPolicy", env, verbose=1, seed=config["seed"], learning_starts=config["learning_starts"],
                    gamma=config["gamma"], learning_rate=config["lr"], tensorboard_log=tb_path,
                    batch_size=config["batch_size"], device='cpu')

    eval_callback = EvalCallback(env, n_eval_episodes=5,eval_freq=500)

    # do not reset num timesteps when continuing learning, this will keep the logging consistent between runs.
    model.learn(total_timesteps=config["time_steps"], log_interval=5, callback=[eval_callback])
