import logging
from pathlib import Path

import wandb
from pybullet_sim.pick_env import UR3ePick

from robot_learning.se2_action_map_q_learning import SpatialActionDQN, seed_all

if __name__ == "__main__":

    config = {
        "n_demonstration_steps": 100,
        "lr": 3e-4,
        "batch_size": 4,
        "n_rotations": 1,
        "n_channels": 64,
        "n_downsampling_layers": 1,
        "n_resnet_blocks": 1,
        "discount_factor": 0.0,
    }
    seed_all(2022)
    logging.basicConfig(level=logging.INFO)
    wandb.init(project="spatial-action-pybullet-pick", dir=str(Path(__file__).parents[1]), mode="online")
    env = UR3ePick(use_motion_primitive=True, use_spatial_action_map=True, simulate_realtime=False)
    dqn = SpatialActionDQN(env.image_dimensions[0], device="cuda", **config)
    dqn.train(env, 1000)
