import torch
from pathlib import Path


class Impala:
    def __init__(self, model, optimizer_maker, use_cuda):
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            self.transfer_stream = torch.cuda.Stream(device=self.device.index)
        else:
            self.device = torch.device("cpu")
            self.transfer_stream = None

        self.model = model.to(self.device)
        self.optimizer = optimizer_maker(self.model.parameters())

        self.clip_rho_threshold = torch.Tensor([1.0]).to(self.device)
        self.clip_c_threshold = torch.Tensor([1.0]).to(self.device)
        self.prev_operation = None

    def predict_impl(self, observations, policies_out):
        self.model.eval()
        with torch.no_grad():
            probs = self.model.probs(observations)
            torch.from_numpy(policies_out).copy_(probs)

    def calc_vs_and_pg_advantages(self, probs, values, actions, rewards, behaviour_policies,
                                  discounts):
        t_max = actions.shape[0]
        batch_size = actions.shape[1]

        target_policies = probs.gather(2, actions)
        policy_ratio = target_policies / behaviour_policies
        rhos = torch.min(policy_ratio, self.clip_rho_threshold)
        cs = torch.min(policy_ratio, self.clip_c_threshold)
        deltas = rhos * (rewards + discounts * values[1:] - values[:t_max])

        for i in reversed(range(0, t_max - 1)):
            deltas[i].add_(discounts[i] * cs[i] * deltas[i + 1])

        vs = deltas + values[:t_max]

        pg_advantages = torch.empty((t_max, batch_size, 1), device=self.device)
        torch.mul(rhos[:t_max - 1], rewards[:t_max - 1] + discounts[:t_max - 1]
                  * vs[1:] - values[:t_max - 1], out=pg_advantages[:t_max - 1])
        pg_advantages[t_max - 1].copy_(deltas[t_max - 1])

        return vs.detach(), pg_advantages.detach()

    def calc_loss(self, log_probs, probs, values, actions, vs, pg_advantages, loss_coefs,
                  data_size):
        v_loss = 0.5 * ((values - vs).pow(2) * loss_coefs).sum() / data_size
        pi_loss = -((log_probs.gather(2, actions) * pg_advantages) * loss_coefs).sum() / data_size
        entropy_loss = ((log_probs * probs).sum(2, keepdim=True) * loss_coefs).sum() / data_size
        return v_loss, pi_loss, entropy_loss

    def train_impl(self, observations, actions, rewards, behaviour_policies, discounts, loss_coefs,
                   data_size):
        self.model.train()
        t_max = actions.shape[0]
        batch_size = actions.shape[1]

        self.optimizer.zero_grad()
        hidden = self.model.convert_obs_to_hidden(observations)
        probs, log_probs = self.model.probs_and_log_probs_from_hidden(
            self.model.slice_hidden(hidden, t_max * batch_size))
        probs = probs.reshape(t_max, batch_size, -1)
        log_probs = log_probs.reshape(t_max, batch_size, -1)
        values = self.model.v_from_hidden(hidden)
        values = values.reshape(t_max + 1, batch_size, 1)
        with torch.no_grad():
            vs, pg_advantages = self.calc_vs_and_pg_advantages(
                probs.detach(), values.detach(), actions, rewards, behaviour_policies, discounts)
        v_loss, pi_loss, entropy_loss = self.calc_loss(
            log_probs, probs, values[:t_max], actions, vs, pg_advantages, loss_coefs, data_size)
        loss = (0.5 * v_loss + pi_loss + 1e-3 * entropy_loss)
        loss.backward()
        self.optimizer.step()
        return v_loss.item(), pi_loss.item(), entropy_loss.item()

    def predict(self, observations_in, policies_out):
        if self.use_cuda:
            self.transfer_stream.synchronize()
            with torch.cuda.stream(self.transfer_stream):
                observations = self.model.convert_obs_to_tensor(observations_in, self.device)
        else:
            observations = self.model.convert_obs_to_tensor(observations_in, self.device)

        def predict_operation():
            return self.predict_impl(observations, policies_out)

        operation = self.prev_operation
        self.prev_operation = predict_operation
        if operation is not None:
            return operation()
        return None

    def train(self, observations_in, actions_in, rewards_in, behaviour_policies_in, discounts_in,
              loss_coefs_in, data_sizes):
        if self.use_cuda:
            self.transfer_stream.synchronize()
            with torch.cuda.stream(self.transfer_stream):
                observations = self.model.convert_obs_to_tensor(observations_in, self.device)
                actions = torch.from_numpy(actions_in).to(device=self.device, non_blocking=True)
                rewards = torch.from_numpy(rewards_in).to(device=self.device, non_blocking=True)
                behaviour_policies = torch.from_numpy(behaviour_policies_in).to(
                    device=self.device, non_blocking=True)
                discounts = torch.from_numpy(discounts_in).to(device=self.device, non_blocking=True)
                loss_coefs = torch.from_numpy(loss_coefs_in).to(
                    device=self.device, non_blocking=True)
        else:
            observations = self.model.convert_obs_to_tensor(observations_in, self.device)
            actions = torch.from_numpy(actions_in)
            rewards = torch.from_numpy(rewards_in)
            behaviour_policies = torch.from_numpy(behaviour_policies_in)
            discounts = torch.from_numpy(discounts_in)
            loss_coefs = torch.from_numpy(loss_coefs_in)

        def train_operation():
            return self.train_impl(observations, actions, rewards, behaviour_policies, discounts,
                                   loss_coefs, sum(data_sizes))

        operation = self.prev_operation
        self.prev_operation = train_operation
        if operation is not None:
            return operation()
        return None

    def sync(self):
        if self.use_cuda:
            self.transfer_stream.synchronize()
        operation = self.prev_operation
        self.prev_operation = None
        if operation is not None:
            return operation()
        return None

    def save_model(self, index):
        output_dir = Path(f"output/{index}").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / "model.pth")
        torch.save(self.optimizer.state_dict(), output_dir / "optimizer.pth")

    def load_model(self, index):
        model_dir = Path(f"output/{index}").resolve()
        self.model.load_state_dict(torch.load(model_dir / "model.pth", map_location="cpu"))
        self.optimizer.load_state_dict(torch.load(model_dir / "optimizer.pth", map_location="cpu"))
