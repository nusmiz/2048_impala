import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model


class G2048A3CModel(Model):
    def __init__(self):
        super(G2048A3CModel, self).__init__()
        self.l_1 = nn.Linear(288, 512)
        self.l_2 = nn.Linear(512, 512)
        self.l_3 = nn.Linear(96, 128)
        self.l_4 = nn.Linear(128, 64)
        self.l_5 = nn.Linear(960, 512)
        self.l_6 = nn.Linear(768, 512)
        self.conv_1 = nn.Conv1d(64, 64, 3)
        self.conv_2 = nn.Conv1d(64, 64, 2)
        self.l_pi = nn.Linear(1536, 4)
        self.l_v = nn.Linear(1536, 1)

        self.rot_matrix = None

    def convert_obs_to_hidden(self, observation):
        h0 = observation[0].reshape(-1, 8, 288)
        h0 = F.leaky_relu(self.l_1(h0))
        h0 = F.leaky_relu(self.l_2(h0))
        h1 = observation[1].reshape(-1, 8, 15, 96)
        h1 = F.leaky_relu(self.l_3(h1))
        h1 = F.leaky_relu(self.l_4(h1))
        h2 = h1.reshape(-1, 15, 64).permute(0, 2, 1)
        h2 = F.leaky_relu(self.conv_1(h2))
        h2 = F.leaky_relu(self.conv_2(h2))
        h1 = h1.reshape(-1, 8, 960)
        h1 = F.leaky_relu(self.l_5(h1))
        h2 = h2.reshape(-1, 8, 768)
        h2 = F.leaky_relu(self.l_6(h2))
        return torch.cat((h0, h1, h2), dim=2)

    def pi_from_hidden(self, hidden):
        if self.rot_matrix is None:
            self.rot_matrix = torch.Tensor([
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ]).to(hidden.device)
        pi = self.l_pi(hidden)
        pi = torch.matmul(self.rot_matrix, pi.reshape(-1, 8, 4, 1))
        return torch.sum(pi, dim=1).reshape(-1, 4)

    def v_from_hidden(self, hidden):
        v = self.l_v(hidden)
        return torch.sum(v, dim=1)

    def convert_obs_to_tensor(self, observation, device):
        return (torch.from_numpy(observation[0]).to(device=device, non_blocking=True),
                torch.from_numpy(observation[1]).to(device=device, non_blocking=True))
