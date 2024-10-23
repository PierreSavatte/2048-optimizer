import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DQN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.input = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: Tensor):
        x = F.relu(self.input(x))
        x = F.relu(self.layer2(x))
        return self.output(x)
