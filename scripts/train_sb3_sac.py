import pathlib

import gym
import numpy as np
import torch
import torch.nn as nn
import wandb
from pybullet_sim.pick_env import UR3ePick
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wandb.integration.sb3 import WandbCallback

from robot_learning.gym_video_wrapper import VideoRecorderWrapper


class RGBDCnn(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.Space, n_channels_in: int, features_dim: int = 256):
        super(RGBDCnn, self).__init__(observation_space, features_dim)
        n_channels = 32
        self.n_channels_in = n_channels_in
        self.cnn = torch.nn.Sequential(
            nn.Conv2d(n_channels_in, n_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding="same"),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding="same"),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # input is HxWxC but already scaled to 0-1 (for the image).
        assert x.shape[-1] == self.n_channels_in
        x = x.permute(0, 3, 1, 2)
        return self.cnn(x)


if __name__ == "__main__":

    folder_path = pathlib.Path(__file__).parent
    config = {
        "gamma": 0.9,
        "lr": 6e-4,
        "learning_starts": 150,
        "batch_size": 64,
        "time_steps": 20000,
        "seed": 2022,
    }

    torch.manual_seed(config["seed"])
    import pybullet as p

    env = UR3ePick(
        use_spatial_action_map=False, use_motion_primitive=True, simulate_realtime=False, pybullet_mode=p.DIRECT
    )
    env = VideoRecorderWrapper(env, folder_path / "videos")
    env.rescale_factor = 1
    policy_kwargs = dict(
        features_extractor_class=RGBDCnn,
        features_extractor_kwargs=dict(features_dim=128, n_channels_in=4),
    )
    tb_path = folder_path / "ur_pusher_state_tb/"
    # https://docs.wandb.ai/guides/integrations/other/stable-baselines-3
    device = "cuda"
    run = wandb.init(project="spatial-action-pybullet-pick", config=config, sync_tensorboard=True, mode="online")
    model = SAC(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config["seed"],
        learning_starts=config["learning_starts"],
        gamma=config["gamma"],
        learning_rate=config["lr"],
        buffer_size=10000,
        tensorboard_log=tb_path,
        batch_size=config["batch_size"],
        device=device,
        tau=0.02,
        action_noise=NormalActionNoise(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.05, 0.05, 0.02, 0.5])),
    )

    eval_callback = EvalCallback(env, n_eval_episodes=5, eval_freq=500)
    wandb_callback = WandbCallback()
    # do not reset num timesteps when continuing learning, this will keep the logging consistent between runs.
    model.learn(total_timesteps=config["time_steps"], log_interval=5, callback=[wandb_callback])
