import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model


class G2048A3CModel(Model):
    def __init__(self):
        super(G2048A3CModel, self).__init__()
        self.l_1 = nn.Linear(2304, 512)
        self.l_pi = nn.Linear(512, 4)
        self.l_v = nn.Linear(512, 1)

    def convert_obs_to_hidden(self, observation):
        h = observation[0].reshape(-1, 2304)
        h = F.leaky_relu(self.l_1(h))
        return h

    def pi_from_hidden(self, hidden):
        return self.l_pi(hidden)

    def v_from_hidden(self, hidden):
        return self.l_v(hidden)

    def convert_obs_to_tensor(self, observation, device):
        return (torch.from_numpy(observation[0]).to(device=device, non_blocking=True),
                torch.from_numpy(observation[1]).to(device=device, non_blocking=True))
