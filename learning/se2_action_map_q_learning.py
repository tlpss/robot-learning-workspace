import logging
import math
import random
import torch.nn as nn
import numpy as np
import torch
import torchvision.transforms.functional
from typing import Tuple, List

logger = logging.getLogger(__name__)
ActionType =  np.ndarray

class ReplayBuffer(object):
    """Buffer to store environment transitions. Taken from https://github.com/denisyarats/pytorch_sac_ae/blob/master/utils.py """
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
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4,1,5,padding="same")
        self.relu = nn.ReLU()

    def forward(self,x):
        assert len(x.shape) == 4 # BxCxHxW
        # do softmax over the last 2 dimensions
        x = self.relu(self.conv1(x))
        assert x.shape[1] == 1
        return x #Bx1xHxW
class SpatialQNetwork(nn.Module):
    def __init__(self,n_rotations = 1, device= 'cpu') -> None:
        super().__init__()
        self.device = device

        self.q_network = Unet().to(self.device)
        self.n_rotations = n_rotations
        self.rotations_in_radians = np.arange(n_rotations) / n_rotations * np.pi # symmetric gripper! 
        self.pad_size = None
    

    def forward(self, img):
        assert len(img.shape) ==3
        #do rotations
        padded_img = self._pad_for_rotations(img)
        batch_of_rotations = self._get_batch_of_rotations(padded_img).to(self.device)
        #pass batch through network
        rotated_q_maps = self.q_network(batch_of_rotations)
        q_maps = self._rotate_batch_back(rotated_q_maps)
        q_maps = self._unpad_after_rotations(q_maps)
        assert len(q_maps.shape) == 4
        return q_maps.squeeze(1) # B,H,W as it has only one channel!

    def get_action(self, img) -> np.ndarray:
        """Get optimal determinstic action according to Q network

        Args:
            img (_type_): _description_

        Returns:
            ActionType: _description_
        """
        with torch.no_grad():
            q_maps = self.forward(img).detach().cpu().numpy()
        indices = np.unravel_index(np.argmax(q_maps),q_maps.shape)
        theta  = self.rotations_in_radians[indices[0]]
        return np.array([indices[1], indices[2],theta])
        

    def sample_action(self, img, exploration_greedy_epsilon = 0.1) -> ActionType:
    
        if np.random.rand() < exploration_greedy_epsilon:
            action = np.array([random.randint(0,img.shape[-2] -1),random.randint(0,img.shape[-1] -1),random.choice(self.rotations_in_radians)])
        else:
            action = self.get_action(img)
            action[:2] += np.random.randn(2) * 6 # additional exploration noise?
            action[:2] = np.clip(action[:2],0,img.shape[1]-1) # assume square images!
        return action

    def _pad_for_rotations(self,img: torch.Tensor) -> torch.Tensor:
        """

        Args:
            img (torch.Tensor): C x H x W Tensor

        Returns:
            torch.Tensor: nR x C x H x W
        """
        assert len(img.shape) == 3 # no batch!
        if self.pad_size is None:
            self.pad_size =int((np.sqrt(img.shape[1]**2 + img.shape[2]**2) - min(img.shape[1:]))/2)
        padded_img =torchvision.transforms.functional.pad(img,self.pad_size)
        return padded_img

    def _unpad_after_rotations(self, batch: torch.Tensor) -> torch.Tensor:
        assert len(batch.shape) == 4
        unpadded_batch = batch[:,:,self.pad_size:-self.pad_size,self.pad_size:-self.pad_size]
        return unpadded_batch

    def _get_batch_of_rotations(self, img: torch.Tensor):
        assert len(img.shape) == 3
        assert img.device.type == 'cpu'
        rotated_imgs = []
        for rotation in self.rotations_in_radians:
            rot_in_deg = np.rad2deg(rotation)
            rotated_imgs.append(torchvision.transforms.functional.rotate(img,rot_in_deg))
        return torch.stack(rotated_imgs)

    def _rotate_batch_back(self, batch: torch.Tensor):
        assert len(batch.shape)==4
        for i in range(batch.shape[0]):
            angle = - np.rad2deg(self.rotations_in_radians[i])
            batch[i] = torchvision.transforms.functional.rotate(batch[i],angle)
        return batch


class SpatialActionDQN(nn.Module):
    def __init__(self, img_resolution:int):
        super().__init__()
        self.q_network = SpatialQNetwork()
        self.target_q_network = SpatialQNetwork()
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer((img_resolution,img_resolution,4),(3,),10000,'cpu')
        self.optim = torch.optim.Adam(self.q_network.parameters(),lr=3e-4)
        self.discount_factor = 0.9
        self.criterion = torch.nn.HuberLoss()

    def train(self,env, n_iterations:int):
        step = 0

        while (step < n_iterations):
            obs = env.reset()
            done = False
            episode_steps = 0
            while not done:
                with torch.no_grad():
                    torch_obs = self._np_image_to_torch(obs)
                    action = self.q_network.sample_action(torch_obs)
                    next_obs, reward, done,_ = env.step(action)
                    logger.debug(f"experience: {action=}, {reward=},{done=}")
                    self.replay_buffer.add(obs,action,reward,next_obs,done)
                    episode_steps += 1
                self.training_step()
    @staticmethod
    def _np_image_to_torch(x):
        x = torch.tensor(x)
        x = x.permute(2,0,1)
        return x
    def training_step(self):
        # get obs, action(u,v,theta), reward, next_obs
        obs,action,reward,next_obs,done = self.replay_buffer.sample(1)
        obs,action,reward,next_obs,done = obs[0],action[0],reward[0],next_obs[0],done[0]
        obs, next_obs = obs.permute(2,0,1), next_obs.permute(2,0,1)

        # compute the Q-values
        q_values = self.q_network(obs)
        # get the Q-value of the action that was taken
        rot_index = np.argmin((self.q_network.rotations_in_radians-action[2].item())**2)
        action_q_value = q_values[rot_index, int(action[0]),int(action[1])]

        # get the max(Q-value) (==value function of that state) of the target network on the next_obs
        future_reward = np.max(self.target_q_network(next_obs).detach().cpu().numpy())

        # compute TD loss

        td_q_value = reward + self.discount_factor * future_reward
        loss = self.criterion(action_q_value, td_q_value)
        # backprop.

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # update target network 
        soft_update_params(self.q_network, self.target_q_network)
        # return loss 
        return loss.detach().cpu().numpy()

def soft_update_params(net:nn.Module, target_net:nn.Module, tau:float = 0.005):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from pybullet_sim.pick_env import UR3ePick
    img = torch.randn((4,100,100))
    qnet = SpatialQNetwork(n_rotations=3,device='cpu')
    with torch.autograd.set_detect_anomaly(True):
        q_vals = qnet(img)
        print(q_vals.shape)
        action = qnet.get_action(img)
        print(action)
        sampled_action = qnet.sample_action(img, 1.0)
        print(sampled_action)
        env = UR3ePick(use_motion_primitive=True, use_spatial_action_map=True,simulate_realtime=False)
        dqn = SpatialActionDQN(env.image_dimensions[0])
        dqn.train(env,100)