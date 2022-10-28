import logging
from pathlib import Path

import wandb
from pybullet_sim.pick_env import UR3ePick

from robot_learning.spatial_action_map_q_learning import SpatialActionDQN, seed_all

if __name__ == "__main__":

    config = {
        "n_demonstrations": 10,
        "lr": 5e-4,
        "batch_size": 32,
        "n_rotations": 2,
        "n_channels": 64,
        "n_downsampling_layers": 2,
        "n_resnet_blocks": 6,
        "discount_factor": 0,
        "action_sample_temperature": 0.0,
        "n_demo_training_steps": 0,
        "n_training_steps": 5000,
        "seed": 2022,
    }
    logging.basicConfig(level=logging.INFO)
    wandb.init(
        project="spatial-action-pybullet-pick", dir=str(Path(__file__).parents[1]), mode="online", config=config
    )
    # get possibly updated config from wandb
    config = wandb.config

    seed_all(config["seed"])
    env = UR3ePick(use_motion_primitive=True, use_spatial_action_map=True, simulate_realtime=False)
    dqn = SpatialActionDQN(env.image_dimensions[0], device="cuda", **config)

    n_demonstrations = config["n_demonstrations"]
    n_demo_training_steps = config["n_demo_training_steps"]
    if n_demonstrations > 0 and n_demo_training_steps > 0:
        print("--- adding demonstrations to buffer")
        demo_path = Path(__file__).parents[1] / "demonstrations" / "pick_env_spatial_actions_demos"
        dqn.add_demonstrations_to_buffer(demo_path, n_demonstrations)
        print("--- training on demonstrations")
        dqn.train_on_buffer(n_demo_training_steps)
    print("--- collecting data and training")
    dqn.collect_and_train(env, config["n_training_steps"])
