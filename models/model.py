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
        if isinstance(observation, np.ndarray):
            return torch.from_numpy(observation).to(device=device, non_blocking=True)
        if isinstance(observation, torch.Tensor):
            return observation.to(device=device, non_blocking=True)
        if isinstance(observation, tuple):
            return tuple(self.convert_obs_to_tensor(obs, device) for obs in observation)
        if isinstance(observation, list):
            return [self.convert_obs_to_tensor(obs, device) for obs in observation]
        return observation

    def slice_hidden(self, hidden, length):
        if isinstance(hidden, torch.Tensor):
            return hidden[:length]
        if isinstance(hidden, tuple):
            return tuple(self.slice_hidden(h, length) for h in hidden)
        if isinstance(hidden, list):
            return [self.slice_hidden(h, length) for h in hidden]
        raise NotImplementedError()
