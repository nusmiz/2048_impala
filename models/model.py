import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def convert_obs_to_hidden(self, observation):
        raise NotImplementedError()

    def forward(self, observation):
        hidden = self.convert_obs_to_hidden(observation)
        return self.pi_from_hidden(hidden), self.v_from_hidden(hidden)

    def pi(self, observation):
        hidden = self.convert_obs_to_hidden(observation)
        return self.pi_from_hidden(hidden)

    def v(self, observation):
        hidden = self.convert_obs_to_hidden(observation)
        return self.v_from_hidden(hidden)

    def pi_from_hidden(self, hidden):
        raise NotImplementedError()

    def v_from_hidden(self, hidden):
        raise NotImplementedError()

    def convert_obs_to_tensor(self, observation, device):
        return torch.from_numpy(observation).to(device=device, non_blocking=True)

    def slice_hidden(self, hidden, length):
        return hidden[:length]
