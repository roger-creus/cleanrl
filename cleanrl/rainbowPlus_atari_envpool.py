# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import math
import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from IPython import embed

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )
        
class NoisyLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        std_init: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @property
    def weight(self):
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self):
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                return self.bias_mu
        else:
            return None

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, channels, size=84):
        super().__init__()
        self.conv0 = layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
        self.layer_norm0 = nn.LayerNorm([channels, size, size])

        self.conv1 = layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
        self.layer_norm1 = nn.LayerNorm([channels, size, size])


    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = self.layer_norm0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        x = self.layer_norm1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, size=84):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = layer_init(nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1))
        self.layer_norm = nn.LayerNorm([self._out_channels, self._input_shape[1], self._input_shape[2]])
        self.res_block0 = ResidualBlock(self._out_channels, (size+1) // 2)
        self.res_block1 = ResidualBlock(self._out_channels, (size+1) // 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class DuelingDistributionalNoisyQNetwork(nn.Module):
    def __init__(self, env, n_envs=128, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n

        c, h, w = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels, size=h)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
            h = shape[1]

        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        
        self.advantage = nn.Sequential(
            NoisyLinear(in_features=shape[0] * shape[1] * shape[2], out_features=512),
            nn.LayerNorm(512),
            nn.ReLU(),
            NoisyLinear(512, self.n * n_atoms),
        )

        self.value = nn.Sequential(
            NoisyLinear(in_features=shape[0] * shape[1] * shape[2], out_features=512),
            nn.LayerNorm(512),
            nn.ReLU(),
            NoisyLinear(512, n_atoms),
        )
        
    def forward(self, x, action=None):
        # dueling
        x = self.network(x / 255.0)
        advantage = self.advantage(x).view(-1, self.n, self.n_atoms)
        value = self.value(x).view(-1, 1, self.n_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # distributional
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]
        
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        img_width=84,
        img_height=84,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = DuelingDistributionalNoisyQNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    print(q_network)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.minibatch_size)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    pmfs = torch.zeros((args.num_steps, args.num_envs, args.n_atoms)).to(device)
    atoms = torch.zeros((args.num_steps, args.n_atoms)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            #epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            #random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,)).to(device)
            with torch.no_grad():
                action, next_pmfs = q_network(next_obs)
                pmfs[step] = next_pmfs
                atoms[step] = q_network.atoms

            #explore = (torch.rand((args.num_envs,)).to(device) < epsilon)
            #action = torch.where(explore, random_actions, max_actions)
            #actions[step] = action
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # Compute Q(lambda) targets
        with torch.no_grad():
            ls = torch.zeros((args.num_steps, args.num_envs, args.n_atoms)).to(device)
            us = torch.zeros((args.num_steps, args.num_envs, args.n_atoms)).to(device)
            dmls = torch.zeros((args.num_steps, args.num_envs, args.n_atoms)).to(device)
            dmus = torch.zeros((args.num_steps, args.num_envs, args.n_atoms)).to(device)
            
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    _, last_pmfs = q_network(next_obs)
                    nextnonterminal = 1.0 - next_done
                    last_atoms = rewards[t].unsqueeze(-1) + args.gamma * q_network.atoms * nextnonterminal.unsqueeze(-1)
                    delta_z = q_network.atoms[1] - q_network.atoms[0]
                    
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    last_pmfs = pmfs[t + 1, :, :]
                    last_atoms = rewards[t].unsqueeze(-1) + args.gamma * (
                        args.q_lambda * last_atoms + (1 - args.q_lambda) * atoms[t + 1, :] * nextnonterminal.unsqueeze(-1)
                    )
                    delta_z = atoms[t + 1, 1] - atoms[t + 1, 0]
                
                tz = last_atoms.clamp(args.v_min, args.v_max)
                b = (tz - args.v_min) / delta_z.unsqueeze(-1)
                l = b.floor().clamp(0, args.n_atoms - 1)
                u = b.ceil().clamp(0, args.n_atoms - 1)
                d_m_l = (u + (l == u).float() - b) * last_pmfs
                d_m_u = (b - l) * last_pmfs
                
                ls[t] = l
                us[t] = u
                dmls[t] = d_m_l
                dmus[t] = d_m_u

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dmls = dmls.reshape(-1, args.n_atoms)
        b_dmus = dmus.reshape(-1, args.n_atoms)
        b_ls = ls.reshape(-1, args.n_atoms)
        b_us = us.reshape(-1, args.n_atoms)

        # Optimizing the Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                target_pmfs = torch.zeros_like(b_dmus[mb_inds])
                mb_b_ls = b_ls[mb_inds]
                mb_b_us = b_us[mb_inds]
                mb_b_dmls = b_dmls[mb_inds]
                mb_b_dmus = b_dmus[mb_inds]
                
                for i in range(target_pmfs.size(0)):
                    target_pmfs[i].index_add_(0, mb_b_ls[i].long(), mb_b_dmls[i])
                    target_pmfs[i].index_add_(0, mb_b_us[i].long(), mb_b_dmus[i])

                _, old_pmfs = q_network(b_obs[mb_inds], b_actions[mb_inds].long())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/td_loss", loss, global_step)
        
        old_val = (old_pmfs * q_network.atoms).sum(1)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
