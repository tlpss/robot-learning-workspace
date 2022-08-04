from typing import List

from ur_sim.demonstrations import Demonstration
from ur_sim.push_env import UR3ePush
from pathlib import Path

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss
import numpy as np

def collect_demos():
    env = UR3ePush(state_observation=True, push_primitive=False,real_time=False)
    demos = env.collect_demonstrations(20,str(Path(__file__).parent / "push_demos.pkl"))
    return demos
def collect_dummy_demos(n_samples):
    x = np.random.rand(4*n_samples).reshape((n_samples,-1))
    y = np.zeros((n_samples,2))
    # dummy actions made up of difference of inputs (e.g. point mass target)
    y[:,0] = x[:,1] - x[:,0]
    y[:,1] = x[:,2] - x[:,3]

    random_episode_size = 21
    x = np.array_split(x,random_episode_size)
    y = np.array_split(y,random_episode_size)
    demonstrations = []
    for i in range(len(x)):
        demonstration = Demonstration()
        demonstration.observations = x[i].tolist()
        demonstration.actions = y[i][:-1].tolist() # final obs has no action!

        demonstrations.append(demonstration)
    return demonstrations




class DemonstrationDataset(Dataset):
    def __init__(self, demonstrations: List[Demonstration]):
        super().__init__()
        self.demonstrations = demonstrations

        # preprocess from rollout to single list
        self.observations = []
        self.actions = []
        for demonstration in self.demonstrations:
            self.observations.extend(demonstration.observations[:-1]) # remove final observation
            self.actions.extend(demonstration.actions)
        self.observations = torch.tensor(self.observations)
        self.actions = torch.tensor(self.actions)
    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        #todo: handle images

        obs = self.observations[idx]
        action = self.actions[idx]

        return obs,action


class BC(pl.LightningModule):
    def __init__(self, agent, lr):
        super().__init__()
        self.agent = agent
        self.lr = lr
        self.criterion = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

    def forward(self, x):
        return self.agent(x)

    def _shared_step(self, batch,idx):
        observations, actions = batch
        predicted_actions = self.agent(observations)
        loss = self.criterion(predicted_actions, actions)
        return loss

    def training_step(self, train_batch, idx):
        loss = self._shared_step(train_batch,idx)
        self.log("train/loss",loss,prog_bar=True)
        return loss

    def validation_step(self, val_batch,idx):
        loss = self._shared_step(val_batch,idx)
        self.log("val/loss",loss,prog_bar=True)
        return loss


if __name__ == "__main__":
    batch_size = 32
    lr = 1e-4
    epochs = 20
    train_demos, val_demos = collect_dummy_demos(10000), collect_dummy_demos(200)
    train_set, val_set = DemonstrationDataset(train_demos), DemonstrationDataset(val_demos)
    train_loader, val_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True), DataLoader(val_set,batch_size=batch_size, shuffle=False)

    print(len(train_set))
    print(train_set[0])

    input_dim = train_set[0][0].shape[0]
    output_dim = train_set[0][1].shape[0]
    agent = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim),
        nn.Tanh()
    )
    bc = BC(agent,lr)

    trainer = pl.Trainer(max_epochs=epochs,log_every_n_steps=1)
    trainer.fit(bc,train_loader,val_loader)


