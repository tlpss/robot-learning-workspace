import logging
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional
from pybullet_sim.pick_env import UR3ePick

from robot_learning.unet import Unet

logger = logging.getLogger(__name__)
ActionType = np.ndarray
import math

import tqdm
import wandb


class ReplayBuffer(object):
    """Buffer to store environment transitions. Taken from https://github.com/denisyarats/pytorch_sac_ae/blob/master/utils.py"""

    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.float16

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones


class SpatialQNetwork(nn.Module):
    def __init__(self, n_rotations=1, n_downsampling_layers=1, n_resnet_blocks=3, n_channels=64, device="cpu") -> None:
        super().__init__()
        self.device = device

        self.backbone = Unet(4, n_downsampling_layers, n_resnet_blocks, n_channels).to(self.device)

        # have to bring head layers to device individually. Cannot move sequential to device..
        self.head = (
            nn.Sequential(  # inspired on https://github.com/andyzeng/visual-pushing-grasping/blob/master/models.py
                nn.Conv2d(n_channels, n_channels, 1, bias=False, device=device),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, 1, 1, bias=False, device=device),
                nn.ReLU(inplace=True),
            )
        )

        self.q_network = nn.Sequential(self.backbone, self.head)
        self.n_rotations = n_rotations
        self.rotations_in_radians = np.arange(n_rotations) / n_rotations * np.pi  # symmetric gripper!
        self.pad_size = None

    def forward(self, img):
        assert len(img.shape) == 3
        # do rotations
        padded_img = self._pad_for_rotations(img)
        batch_of_rotations = self._get_batch_of_rotations(padded_img).to(self.device)
        # pass batch through network
        rotated_q_maps = self.q_network(batch_of_rotations)
        q_maps = self._rotate_batch_back(rotated_q_maps)
        q_maps = self._unpad_after_rotations(q_maps)
        assert len(q_maps.shape) == 4
        return q_maps.squeeze(1)  # B,H,W as it has only one channel!

    def get_action(self, img) -> np.ndarray:
        """Get optimal determinstic action according to Q network

        Args:
            img (_type_): _description_

        Returns:
            ActionType: _description_
        """
        with torch.no_grad():
            q_maps = self.forward(img).detach().cpu().numpy()
        indices = np.unravel_index(np.argmax(q_maps), q_maps.shape)
        theta = self.rotations_in_radians[indices[0]]
        return np.array([indices[2], indices[1], theta]), q_maps

    def sample_action(self, img, exploration_greedy_epsilon=0.1, temperature=1e-8) -> ActionType:

        if np.random.rand() < exploration_greedy_epsilon:
            action = np.array(
                [
                    random.randint(0, img.shape[2] - 1),
                    random.randint(0, img.shape[1] - 1),
                    random.choice(self.rotations_in_radians),
                ]
            )
            q_maps = None
        else:
            # sample an action according to the heatmaps
            with torch.no_grad():
                q_maps = self.forward(img).detach().cpu().numpy()
                probability_map = np.copy(q_maps)
                probability_map += temperature
                probability_map /= np.sum(probability_map)
                flattened_probabilities = probability_map.flatten()

                sampled_flattened_index = np.random.choice(
                    np.arange(len(flattened_probabilities)), p=flattened_probabilities
                )
                sampled_indices = np.unravel_index(sampled_flattened_index, probability_map.shape)
                theta = self.rotations_in_radians[sampled_indices[0]]
                action = np.array([sampled_indices[2], sampled_indices[1], theta])
        return action, q_maps

    def _pad_for_rotations(self, img: torch.Tensor) -> torch.Tensor:
        """

        Args:
            img (torch.Tensor): C x H x W Tensor

        Returns:
            torch.Tensor: nR x C x H x W
        """
        assert len(img.shape) == 3  # no batch!
        if self.pad_size is None:
            self.pad_size = int((np.sqrt(img.shape[1] ** 2 + img.shape[2] ** 2) - min(img.shape[1:])) / 2)
            self.pad_size = math.ceil(self.pad_size / 16) * 16  # make sure image remains multiple of 32.
            print(f"padding size = {self.pad_size}")
        padded_img = torchvision.transforms.functional.pad(img, self.pad_size)
        return padded_img

    def _unpad_after_rotations(self, batch: torch.Tensor) -> torch.Tensor:
        assert len(batch.shape) == 4
        unpadded_batch = batch[:, :, self.pad_size : -self.pad_size, self.pad_size : -self.pad_size]
        return unpadded_batch

    def _get_batch_of_rotations(self, img: torch.Tensor):
        assert len(img.shape) == 3
        rotated_imgs = []
        for rotation in self.rotations_in_radians:
            rot_in_deg = np.rad2deg(rotation)
            rotated_imgs.append(torchvision.transforms.functional.rotate(img, rot_in_deg))
        return torch.stack(rotated_imgs)

    def _rotate_batch_back(self, batch: torch.Tensor):
        assert len(batch.shape) == 4
        rotated = []
        for i in range(batch.shape[0]):
            angle = -np.rad2deg(self.rotations_in_radians[i])
            rotated.append(torchvision.transforms.functional.rotate(batch[i], angle))
        return torch.stack(rotated)


class SpatialActionDQN(nn.Module):
    def __init__(
        self,
        img_resolution: int,
        discount_factor=0.0,
        action_sample_temperature=1e-6,
        n_rotations=1,
        n_resnet_blocks=3,
        n_downsampling_layers=1,
        n_channels=64,
        batch_size: int = 3,
        n_demonstration_steps: int = 100,
        lr: float = 3e-4,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.reactive_policy = True
        self.device = device

        self.discount_factor = discount_factor

        self.start_training_step = 51
        self.log_every_n_steps = 21

        self.n_bootstrap_steps = n_demonstration_steps
        self.criterion = torch.nn.L1Loss()

        self.n_rotations = n_rotations
        self.batch_size = batch_size
        self.action_sample_temperature = action_sample_temperature

        self.q_network = SpatialQNetwork(
            n_rotations, n_downsampling_layers, n_resnet_blocks, n_channels, device=self.device
        )
        if self.discount_factor > 0.0:
            self.target_q_network = SpatialQNetwork(
                n_rotations, n_downsampling_layers, n_resnet_blocks, n_channels, device=self.device
            )
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        else:
            print(
                "discount factor was set to zero, the policy will be completely reactive and not consider future rewards."
            )

        self.replay_buffer = ReplayBuffer((img_resolution, img_resolution, 4), (3,), 10000, self.device)
        self.optim = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.log = wandb.log

    def train(self, env: UR3ePick, n_iterations: int):
        done = True
        for iteration in tqdm.trange(n_iterations):
            if done:
                obs = env.reset()
                done = False
            self.log({"iteration": iteration}, step=iteration)
            with torch.no_grad():
                torch_obs = self._np_image_to_torch(obs)
                if iteration > self.n_bootstrap_steps:
                    action, q_maps = self.q_network.sample_action(
                        torch_obs, exploration_greedy_epsilon=0.1, temperature=self.action_sample_temperature
                    )
                    if iteration % self.log_every_n_steps == 0:
                        if q_maps is not None:
                            for i in range(self.q_network.n_rotations):
                                self.log(
                                    {
                                        f"interaction_spatial_action_map_{self.q_network.rotations_in_radians[i]:.2f}": wandb.Image(
                                            q_maps[i]
                                        )
                                    },
                                    step=iteration,
                                )
                else:
                    action = env.get_oracle_action()

                next_obs, reward, done, _ = env.step(action)

                if iteration % self.log_every_n_steps == 0:
                    self.visualize_action(obs, action, "interaction")
                    self.log({"interaction_next_obs": wandb.Image(next_obs[..., :3])}, step=iteration)
                self.log({"reward": reward}, step=iteration)
                logger.info(f"experience: {action=}, {reward=},{done=}")
                self.replay_buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs
            if iteration > self.start_training_step:
                loss = self.training_step(iteration)
                self.log({"train_loss": loss}, step=iteration)

    def visualize_action(self, obs, action, caption_prefix=""):
        scale = 1 + obs.shape[0] // 64
        obs = obs[..., :3]
        obs = np.ascontiguousarray(obs)
        u, v = int(action[0]), int(action[1])
        theta = action[2]
        vis = cv2.circle(obs, (u, v), scale, (0, 0, 0), -1)
        grasp_orientation_vector = np.array([np.cos(theta), np.sin(theta)])
        line_start = np.array([u, v]) - grasp_orientation_vector * scale * 3
        line_start = np.clip(0, obs.shape[0] - 1, line_start)
        line_end = np.array([u, v]) + grasp_orientation_vector * scale * 3
        line_end = np.clip(0, obs.shape[0] - 1, line_end)
        vis = cv2.line(
            vis,
            line_start.astype(np.uint8),
            line_end.astype(np.uint8),
            (
                0,
                0,
                0,
            ),
            scale // 2,
        )
        self.log({f"{caption_prefix}_action": wandb.Image(vis)})

    @staticmethod
    def _np_image_to_torch(x):
        x = torch.tensor(x)
        x = x.permute(2, 0, 1)
        return x

    def training_step(self, iteration):
        action_q_values = []
        td_q_values = []
        for batch_index in range(self.batch_size):
            # get obs, action(u,v,theta), reward, next_obs
            obs, action, reward, next_obs, done = self.replay_buffer.sample(1)
            obs, action, reward, next_obs, done = obs[0], action[0], reward[0], next_obs[0], done[0]
            obs, next_obs = obs.permute(2, 0, 1), next_obs.permute(2, 0, 1)

            # compute the Q-values
            q_values = self.q_network(obs)
            # get the Q-value of the action that was taken
            rot_index = np.argmin((self.q_network.rotations_in_radians - action[2].item()) ** 2)
            action_q_value = q_values[rot_index, int(action[1]), int(action[0])].unsqueeze(
                0
            )  # make sure dim = (1,) to match td_q_value

            action_q_values.append(action_q_value)

            # compute TD loss
            if self.discount_factor > 0.0:
                # get the max(Q-value) (==value function of that state) of the target network on the next_obs
                future_reward = torch.max(self.target_q_network(next_obs))
                td_q_value = reward + self.discount_factor * future_reward
            else:
                td_q_value = reward

            td_q_values.append(td_q_value)

        if iteration % self.log_every_n_steps == 0:
            for i in range(self.q_network.n_rotations):
                self.log(
                    {
                        f"train_spatial_action_map_{self.q_network.rotations_in_radians[i]:.2f}": wandb.Image(
                            q_values[i].detach().cpu()
                        )
                    }
                )
            self.visualize_action(obs.detach().cpu().permute(1, 2, 0)[..., :3], action.cpu(), caption_prefix="train")

        action_q_values = torch.stack(action_q_values)
        td_q_values = torch.stack(td_q_values)
        loss = self.criterion(action_q_values, td_q_values)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.discount_factor > 0.0:
            # update target network
            soft_update_params(self.q_network, self.target_q_network)

        loss = loss.detach().cpu().numpy()
        return loss


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float = 0.005):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def seed_all(x):
    torch.random.manual_seed(x)
    np.random.seed(x)
    random.seed(x)


if __name__ == "__main__":
    network = SpatialQNetwork(2, device="cuda")
    img = torch.randn(4, 256, 256).to("cuda")
    print(img.device)
    map = network(img)
    print(map.shape)
