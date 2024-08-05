# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
from collections import deque
import warnings
from dataclasses import dataclass, field
from typing import List, Literal
from rich.pretty import pprint
import torch.distributed as dist
import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


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
    local_num_envs: int = 256
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
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""
    device_ids: List[int] = field(default_factory=lambda: [])
    """the device ids that subprocess workers will use"""
    backend: Literal["gloo", "nccl", "mpi"] = "gloo"
    """the backend for distributed training"""

    # to be filled in runtime
    local_batch_size: int = 0
    """the local batch size in the local rank (computed in runtime)"""
    local_minibatch_size: int = 0
    """the local mini-batch size in the local rank (computed in runtime)"""
    num_envs: int = 0
    """the number of parallel game environments (computed in runtime)"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    world_size: int = 0
    """the number of processes (computed in runtime)"""


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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LayerNorm([64, 7, 7]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    # torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py
    # taken from https://pytorch.org/docs/stable/elastic/run.html
    args = tyro.cli(Args)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    args.num_envs = args.local_num_envs * args.world_size
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.world_size > 1:
        dist.init_process_group(args.backend, rank=local_rank, world_size=args.world_size)
    else:
        warnings.warn(
            """
Not using distributed mode!
If you want to use distributed mode, please execute this script with 'torchrun'.
E.g., `torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py`
        """
        )
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = None
    if local_rank == 0:
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
        pprint(args)

    # TRY NOT TO MODIFY: seeding
    # CRUCIAL: note that we needed to pass a different seed for each data parallelism worker
    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - local_rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if len(args.device_ids) > 0:
        assert len(args.device_ids) == args.world_size, "you must specify the same number of device ids as `--nproc_per_node`"
        device = torch.device(f"cuda:{args.device_ids[local_rank]}" if torch.cuda.is_available() and args.cuda else "cpu")
    else:
        device_count = torch.cuda.device_count()
        if device_count < args.world_size:
            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
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

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,)).to(device)
            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            explore = (torch.rand((args.num_envs,)).to(device) < epsilon)
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if not writer:
                continue

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        print(
            f"local_rank: {local_rank}, action.sum(): {action.sum()}, iteration: {iteration}"
        )
        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = rewards[t] + args.gamma * (
                        args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        # Optimizing the Q-network
        b_inds = np.arange(args.local_batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.local_batch_size, args.local_minibatch_size):
                end = start + args.local_minibatch_size
                mb_inds = b_inds[start:end]

                old_val = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                loss = F.mse_loss(b_returns[mb_inds], old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                
                if args.world_size > 1:
                    # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in q_network.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in q_network.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / args.world_size
                            )
                            offset += param.numel()
                            
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        if local_rank == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)
            writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
