from collections import defaultdict
from copy import copy
import numpy as np

# import multiprocessing
# from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import tqdm

# from latpol2img import LatPol2
import torch
import tensordict
from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
import torch.nn as nn
import multiprocessing
import gym
from torchrl.modules.distributions import TanhNormal
from torchrl.envs import (
    GymWrapper,
    StepCounter,
    TransformedEnv,
    InitTracker,
    Compose,
    GymEnv,
    RewardSum,
)
from torchrl.envs import check_env_specs, ToTensorImage, Resize
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    NormalParamExtractor,
    ValueOperator,
    ConvNet,
    LSTMModule,
    ActorCriticOperator,
)
from torchrl.objectives import SACLoss, SoftUpdate
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.collectors import SyncDataCollector
from torchrl._utils import logger as torchrl_logger
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import log_metrics

# Check if GPU (CUDA) access is available
# is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available()  # and not is_fork
    else torch.device("cpu")
)
torchrl_logger.info(f"Running on device: {device}")

exp_name = generate_exp_name("DiscreteSAC", "trl_lstm_test_exp")
logger = get_logger(
    logger_type="wandb",
    logger_name="TRL_Logger",
    experiment_name=exp_name,
    wandb_kwargs={
        # "mode": cfg.logger.mode,
        # "config": dict(cfg),
        "project": "trl_lstm_test",
        # "group": cfg.logger.group_name,
    },
)

# Global configuration variables
buffer_size = 200_000
max_steps = 2e6
lr = 1e-4
init_random_steps = 5_000

# Environment initialization
# WD_THRESH = 3  # [m]
# STATE_RES = (128, 64)
# latenv = LatPol2(wdthresh=WD_THRESH, state_res=STATE_RES, dT=3)

# env = GymWrapper(latenv)
env = GymEnv("Pendulum-v1", from_pixels=True, device=device)
check_env_specs(env)
env = TransformedEnv(
    env,
    Compose(
        ToTensorImage(),
        InitTracker(),
        StepCounter(),
        Resize(
            64, 64
        ),  # I cannot get this to work on the HPC, as torchvision is not working properly
        RewardSum(),
    ),
)

# Networks
conv = ConvNet(
    in_features=3,
    num_cells=[32, 64, 256],
    squeeze_output=True,
    aggregator_class=nn.AdaptiveAvgPool2d,
    aggregator_kwargs={"output_size": (1, 1)},
    device=device,
)
conv_mod = Mod(conv, in_keys=["pixels"], out_keys=["embedding"])

# Get the number of cells in the last layer
n_cells = conv_mod(env.reset().to(device))["embedding"].shape[-1]

lstm = LSTMModule(
    input_size=n_cells,
    hidden_size=256,
    device=device,
    in_key="embedding",
    out_key="embedding",
    python_based=True,
)

# Common feature extractor
feature_extractor = Seq(conv_mod, lstm.set_recurrent_mode())

actor_seq = nn.Sequential(
    nn.Linear(n_cells, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, env.action_spec.shape[-1] * 2),
    NormalParamExtractor(),
).to(device)

actor_module = Mod(actor_seq, in_keys=["embedding"], out_keys=["loc", "scale"])

actor = ProbabilisticActor(
    actor_module,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=TanhNormal,
)


class ValueMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        """
        Initialize the ValueMLP network.

        Args:
            input_dim (int): Number of input features. Defaults to 3, i.e., 2+1.
            hidden_dim (int): Number of neurons in the hidden layers. Defaults to 64.
        """
        super(ValueMLP, self).__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, action):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.cat([x, action], dim=-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


qvalue = ValueOperator(
    ValueMLP().to(device),
    in_keys=["embedding", "action"],
    out_keys=["state_action_value"],
)

ac_operator = ActorCriticOperator(feature_extractor, actor, qvalue)
ac_operator.get_critic_operator()(env.reset().to(device))

# Make policy aware of supplementary inputs and
# outputs during rollout execution.
env.append_transform(lstm.make_tensordict_primer())

loss = SACLoss(
    actor_network=ac_operator.get_policy_operator(),
    qvalue_network=ac_operator.get_critic_operator(),
    loss_function="l2",
    target_entropy=0.7,
)


# Synchronized Data collector
fpb = 256
collector = SyncDataCollector(
    env,
    policy=ac_operator.get_policy_operator(),
    frames_per_batch=fpb,
    total_frames=max_steps,
    init_random_frames=init_random_steps,
    device=device,
)

# Set update rule for the target network
updater = SoftUpdate(loss, eps=0.99)

# Replay Buffer to store transitions
memory = ReplayBuffer(storage=LazyTensorStorage(buffer_size))

# Optimizer
optim = torch.optim.Adam(loss.parameters(), lr, weight_decay=1e-5)

logs = defaultdict(list)
collected_frames = 0

pbar = tqdm.tqdm(total=max_steps)

torchrl_logger.info("Starting trainging")
for i, data in enumerate(collector):
    data: tensordict.TensorDictBase

    memory.extend(data)
    current_frames = data.numel()
    collected_frames += current_frames

    pbar.update(data.numel())

    if len(memory) > init_random_steps:
        avg_length: torch.Tensor = memory[-1000:]["next", "step_count"]
        average_step_count = avg_length.type(torch.float).mean().item()
        sample = memory.sample(128)

        _ = loss.select_out_keys("loss_actor", "loss_qvalue", "loss_alpha")
        losses = loss(sample.to(device))
        tot_loss = losses["loss_actor"] + losses["loss_qvalue"] + losses["loss_alpha"]

        actor_loss, q_loss, alpha_loss = (
            losses["loss_actor"],
            losses["loss_qvalue"],
            losses["loss_alpha"],
        )

        tot_loss.backward()
        optim.step()
        optim.zero_grad()
        updater.step()

        # if i % 50 == 0:
        # Validate the policy
        # with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # eval_rollout = env.rollout(
        # 3000, policy=ac_operator.get_policy_operator()
        # )
        # logs["stepno."].append(i * fpb)
        # print(f"Max action: {eval_rollout['action'].max().item()}")
        # print(f"Min action: {eval_rollout['action'].min().item()}")
        # logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
        # logs["eval reward (sum)"].append(
        # eval_rollout["next", "reward"].sum().item()
        # )
        # logs["average step count"].append(average_step_count)
        # eval_str = (
        # f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
        # f"(init: {logs['eval reward (sum)'][0]: 4.4f}) "
        # f"average step count: {logs['average step count'][-1]: 4.4f} "
        # f"stepno.: {logs['stepno.'][-1]:.0f}"
        # )
        # torchrl_logger.info(eval_str)
        # del eval_rollout

        metrics_to_log = {}
        episode_end = (
            data["next", "done"]
            if data["next", "done"].any()
            else data["next", "truncated"]
        )
        episode_rewards = data["next", "episode_reward"][episode_end]
        if len(episode_rewards) > 0:
            # episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            # metrics_to_log["train/episode_length"] = (
            #     episode_length.sum().item() / len(episode_length)
            # )

        metrics_to_log["train/q_loss"] = q_loss
        metrics_to_log["train/a_loss"] = actor_loss
        metrics_to_log["train/alpha_loss"] = alpha_loss
        # metrics_to_log["train/sampling_time"] = sampling_time
        # metrics_to_log["train/training_time"] = training_time

        log_metrics(logger, metrics_to_log, collected_frames)
        # fig, ax = plt.subplots()
        # ax.plot(
        #     logs["stepno."], logs["eval reward (sum)"], label="Cum. eval reward"
        # )
        # ax.plot(
        #     logs["stepno."],
        #     logs["average step count"],
        #     label="Average step count",
        # )
        # ax.legend()
        # ax.set_xlabel("Step number")
        # plt.savefig("eval_reward.png")
        # plt.close()

