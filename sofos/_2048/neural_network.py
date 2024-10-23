from abc import ABC, abstractmethod

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DQN(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: Tensor): ...


class DQNv1(DQN):

    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.input = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: Tensor):
        x = F.relu(self.input(x))
        x = F.relu(self.layer2(x))
        return self.output(x)


class DQNv2(DQN):

    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2)
        self.layer2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3
        )
        self.output = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: Tensor):
        x = F.relu(self.input(x))
        x = F.relu(self.layer2(x))
        x = x.view(x.size(0), -1)
        return self.output(x)
