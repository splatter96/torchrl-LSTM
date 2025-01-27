# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
from contextlib import nullcontext

import torch
from tensordict.nn import InteractionType, TensorDictModule
from torch._prims_common import get_acc_type
from torchrl.modules.distributions import TanhNormal
from torchrl.envs import check_env_specs, ToTensorImage, Resize

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    CompositeSpec,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    Compose,
    DoubleToFloat,
    InitTracker,
    RewardSum,
    StepCounter,
    TransformedEnv,
    ObservationTransform,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, SafeModule
from torchrl.modules.distributions import OneHotCategorical

from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.objectives import SACLoss, SoftUpdate
from torchrl.record import VideoRecorder

from tensordict.nn import TensorDictSequential
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    NormalParamExtractor,
    ValueOperator,
    ConvNet,
    LSTMModule,
    ActorCriticOperator,
    ActorValueOperator,
    ActorCriticWrapper,
)

from tensordict.utils import (
    NestedKey,
)
from torchrl.data.tensor_specs import TensorSpec
from functools import wraps
from tensordict import TensorDictBase

from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
    check_finite,
)
import torch.nn.functional as F

from copy import copy

from torchrl.data.tensor_specs import ContinuousBox


# ====================================================================
# Environment utils
# -----------------


def _apply_to_composite(function):
    @wraps(function)
    def new_fun(self, observation_spec):
        if isinstance(observation_spec, CompositeSpec):
            _specs = observation_spec._specs
            in_keys = self.in_keys
            out_keys = self.out_keys
            for in_key, out_key in zip(in_keys, out_keys):
                if in_key in observation_spec.keys(True, True):
                    _specs[out_key] = function(self, observation_spec[in_key].clone())
            return CompositeSpec(
                _specs, shape=observation_spec.shape, device=observation_spec.device
            )
        else:
            return function(self, observation_spec)

    return new_fun


class POMDP(ObservationTransform):
    """Turns a pixel observation to grayscale."""

    def __init__(
        self,
        in_keys=None,  #: Sequence[NestedKey] | None = None,
        out_keys=None,  #: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[[0, 2]]

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape

        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


def apply_env_transforms(env, max_episode_steps):
    transformed_env = TransformedEnv(
        env,
        Compose(
            POMDP(),
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def apply_env_transforms_niklas(env, max_episode_steps):
    transformed_env = TransformedEnv(
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
    return transformed_env


def apply_env_transforms_new(env, max_episode_steps):
    transformed_env = TransformedEnv(
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
    return transformed_env


def make_environment(cfg, logger=None):
    """Make environments for training and evaluation."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    train_env = TransformedEnv(GymEnv("CartPole-v1", from_pixels=False, device=device))
    train_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(
        train_env, max_episode_steps=cfg.env.max_episode_steps
    )

    eval_env = TransformedEnv(GymEnv("CartPole-v1", from_pixels=False, device=device))
    eval_env = apply_env_transforms(
        eval_env, max_episode_steps=cfg.env.max_episode_steps
    )

    if cfg.logger.video:
        eval_env = eval_env.insert_transform(
            0, VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


def make_environment_niklas(cfg, logger=None):
    """Make environments for training and evaluation."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    # train_env = TransformedEnv(GymEnv("CartPole-v1", from_pixels=False, device=device))
    train_env = TransformedEnv(GymEnv("Pendulum-v1", from_pixels=True, device=device))
    train_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms_niklas(
        train_env, max_episode_steps=cfg.env.max_episode_steps
    )

    eval_env = TransformedEnv(GymEnv("Pendulum-v1", from_pixels=True, device=device))
    eval_env = apply_env_transforms_niklas(
        eval_env, max_episode_steps=cfg.env.max_episode_steps
    )

    if cfg.logger.video:
        eval_env = eval_env.insert_transform(
            0, VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


def make_environment_new(cfg, logger=None):
    """Make environments for training and evaluation."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    train_env = TransformedEnv(GymEnv("CartPole-v1", from_pixels=True, device=device))
    train_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms_new(
        train_env, max_episode_steps=cfg.env.max_episode_steps
    )

    eval_env = TransformedEnv(GymEnv("CartPole-v1", from_pixels=True, device=device))
    eval_env = apply_env_transforms_new(
        eval_env, max_episode_steps=cfg.env.max_episode_steps
    )

    if cfg.logger.video:
        eval_env = eval_env.insert_transform(
            0, VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        # reset_at_each_iter=cfg.collector.reset_at_each_iter,
        device=device,
        storing_device="cpu",
        # compile_policy=True,
        # cudagraph_policy=True,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    with (
        tempfile.TemporaryDirectory()
        if scratch_dir is None
        else nullcontext(scratch_dir)
    ) as scratch_dir:
        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        return replay_buffer


# ====================================================================
# Model
# -----


def make_sac_agent_original(cfg, train_env, eval_env, device):
    """Make discrete SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    # Define Actor Network
    in_keys = ["observation"]

    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

    actor_module = SafeModule(
        module=actor_net,
        in_keys=in_keys,
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        spec=CompositeSpec(action=eval_env.action_spec),
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }
    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = TensorDictModule(
        in_keys=in_keys,
        out_keys=["action_value"],
        module=qvalue_net,
    )

    model = torch.nn.ModuleList([actor, qvalue]).to(device)
    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model


def make_sac_agent(cfg, train_env, eval_env, device):
    """Make discrete SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    observation_spec = train_env.observation_spec["observation"]
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    # Define Actor Network
    in_keys = ["observation"]

    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

    actor_module = SafeModule(
        module=actor_net,
        in_keys=in_keys,
        out_keys=["logits"],
    ).to(device)

    n_cells = actor_module(train_env.reset())["logits"].shape[-1]

    lstm = LSTMModule(
        # input_size=action_spec.shape[-1],
        input_size=n_cells,
        hidden_size=128,
        device=device,
        in_key="logits",
        out_key="embed",
    )

    lstm_converter_net = MLP(out_features=2, num_cells=[64], device=device)
    lstm_converter = SafeModule(
        module=lstm_converter_net,
        in_keys=["embed"],
        out_keys=["logits"],
    )

    train_env.append_transform(lstm.make_tensordict_primer())
    eval_env.append_transform(lstm.make_tensordict_primer())

    actor1 = TensorDictSequential(
        actor_module, lstm.set_recurrent_mode(True), lstm_converter
    )

    actor = ProbabilisticActor(
        spec=CompositeSpec(action=eval_env.action_spec),
        # module=actor_module,
        module=actor1,
        in_keys=["logits"],
        # in_keys=["embed"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    # qvalue = TensorDictModule(
    #     in_keys=in_keys,
    #     out_keys=["action_value"],
    #     module=MLP(
    #         num_cells=cfg.network.hidden_sizes,
    #         out_features=action_spec.shape[-1],
    #         activation_class=get_activation(cfg),
    #     ),
    # )
    #
    # print("qval old")
    # print(qvalue)

    obs_embedd_size = 32
    action_embedd_size = 8
    reward_embedd_size = 8
    shortcut_embedd_size = 8
    q_hiden_size = 128

    obs_embedder = TensorDictModule(
        in_keys=["observation"],
        out_keys=["obs_embedded"],
        module=MLP(depth=0, out_features=obs_embedd_size, activate_last_layer=True),
    )
    reward_embedder = TensorDictModule(
        in_keys=["reward"],
        out_keys=["reward_embedded"],
        module=MLP(depth=0, out_features=reward_embedd_size, activate_last_layer=True),
    )
    action_embedder = TensorDictModule(
        in_keys=["prev_action"],
        out_keys=["prev_action_embedded"],
        module=MLP(depth=0, out_features=action_embedd_size, activate_last_layer=True),
    )
    hidden_states = CatTensors(
        in_keys=["obs_embedded", "reward_embedded", "prev_action_embedded"],
        out_key="embedding",
    )
    lstm = (
        LSTMModule(
            input_size=obs_embedd_size + action_embedd_size + reward_embedd_size,
            hidden_size=q_hiden_size,
            device=device,
            in_key="embedding",
            out_key="embedding_lstm",
            python_based=True,
        ).set_recurrent_mode(True),
    )
    current_shortcut_embedder = TensorDictModule(
        in_keys=["observation"],
        out_keys=["obs_shortcut_embed"],
        module=MLP(
            depth=0, out_features=shortcut_embedd_size, activate_last_layer=True
        ),
    )
    joint_embed = CatTensors(
        in_keys=["embedding_lstm", "obs_shortcut_embed"],
        out_key="joint_embeds",
    )

    qf = TensorDictModule(
        in_keys=["joint_embeds"],
        out_keys=["action_value"],
        module=MLP(
            num_cells=[q_hiden_size],
            out_features=action_spec.shape[-1],
            activate_last_layer=True,
        ),
    )

    qvalue = TensorDictSequential(
        obs_embedder,
        reward_embedder,
        action_embedder,
        hidden_states,
        lstm,
        current_shortcut_embedder,
        joint_embed,
        qf,
    )

    # print(obs_embedder)
    print(qvalue)
    # exit(0)

    q_hiden_size = 128
    qvalue = TensorDictSequential(
        # First Linear layer
        TensorDictModule(
            in_keys=["observation"],
            out_keys=["obs_embed"],
            # module=MLP(num_cells=[q_hiden_size], out_features=q_hiden_size),
            module=MLP(depth=0, out_features=q_hiden_size, activate_last_layer=True),
        ),
        # LSTM Module in the middle
        LSTMModule(
            input_size=q_hiden_size,
            hidden_size=q_hiden_size,
            device=device,
            in_key="obs_embed",
            out_key="obs_lstm",
            python_based=True,
        ).set_recurrent_mode(True),
        # Second Linear layer
        TensorDictModule(
            in_keys=["obs_lstm"],
            out_keys=["action_value"],
            module=MLP(
                in_features=q_hiden_size,
                num_cells=[q_hiden_size],
                out_features=action_spec.shape[-1],
            ),
        ),
    )

    # print("Q Value network")
    # print(qvalue)
    # exit(0)

    # Combine modules to actor critic model
    model = torch.nn.ModuleList([actor, qvalue]).to(device)
    # init nets
    print("initializing net")
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        # td = eval_env.reset()
        td = train_env.reset()
        td = td.to(device)
        print(td)
        for net in model:
            net(td)
    del td
    eval_env.close()
    print("done initializing net")

    return model


def make_sac_agent_niklas(cfg, train_env, eval_env, device):
    # Networks
    conv = ConvNet(
        in_features=3,
        num_cells=[32, 64, 256],
        squeeze_output=True,
        aggregator_class=nn.AdaptiveAvgPool2d,
        aggregator_kwargs={"output_size": (1, 1)},
        device=device,
    )
    conv_mod = TensorDictModule(conv, in_keys=["pixels"], out_keys=["embedding"])

    # Get the number of cells in the last layer
    n_cells = conv_mod(train_env.reset().to(device))["embedding"].shape[-1]

    lstm = LSTMModule(
        input_size=n_cells,
        hidden_size=256,
        device=device,
        in_key="embedding",
        out_key="embedding",
        python_based=True,
    )

    # Common feature extractor
    feature_extractor = TensorDictSequential(conv_mod, lstm.set_recurrent_mode())

    actor_seq = nn.Sequential(
        nn.Linear(n_cells, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, train_env.action_spec.shape[-1] * 2),
        NormalParamExtractor(),
    ).to(device)

    actor_module = TensorDictModule(
        actor_seq, in_keys=["embedding"], out_keys=["loc", "scale"]
    )

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
    ac_operator.get_critic_operator()(train_env.reset().to(device))

    # Make policy aware of supplementary inputs and
    # outputs during rollout execution.
    train_env.append_transform(lstm.make_tensordict_primer())
    eval_env.append_transform(lstm.make_tensordict_primer())

    # Combine modules to actor critic model
    model = torch.nn.ModuleList(
        [ac_operator.get_policy_operator(), ac_operator.get_critic_operator()]
    ).to(device)
    # init nets
    print("initializing net")
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = train_env.reset()
        td = td.to(device)
        # print(td)
        for net in model:
            res = net(td)
            print(f"After net {res}")
    del td
    eval_env.close()
    print("done initializing net")

    return model


def make_sac_agent_new(cfg, train_env, eval_env, device):
    # Networks
    # conv = ConvNet(
    #     in_features=3,
    #     num_cells=[32, 64, 256],  # TODO make parameters
    #     squeeze_output=True,
    #     aggregator_class=nn.AdaptiveAvgPool2d,
    #     aggregator_kwargs={"output_size": (1, 1)},
    #     device=device,
    # )
    # conv_mod = TensorDictModule(conv, in_keys=["pixels"], out_keys=["embedding"])
    #
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]

    mlp = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=action_spec.shape[-1],
        device=device,
        activation_class=get_activation(cfg),
    )
    # mlp_mod = TensorDictModule(mlp, in_keys=["observation"], out_keys=["embedding"])
    mlp_mod = TensorDictModule(mlp, in_keys=["observation"], out_keys=["embedding"])

    # Get the number of cells in the last layer
    # n_cells = conv_mod(train_env.reset().to(device))["embedding"].shape[-1]
    n_cells = mlp_mod(train_env.reset().to(device))["embedding"].shape[-1]

    lstm = LSTMModule(
        input_size=n_cells,
        hidden_size=256,
        device=device,
        in_key="embedding",
        out_key="embedding",
        python_based=True,
    )

    # Common feature extractor
    # feature_extractor = TensorDictSequential(conv_mod, lstm.set_recurrent_mode())
    feature_extractor = TensorDictSequential(mlp_mod, lstm.set_recurrent_mode())

    # Non LSTM
    #feature_extractor = TensorDictSequential(mlp_mod)

    # TODO replace with MLP??
    # actor_seq = nn.Sequential(
    #     nn.Linear(n_cells, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 64),
    #     nn.ReLU(),
    #     # nn.Linear(64, train_env.action_spec.shape[-1] * 2),
    #     nn.Linear(64, train_env.action_spec.shape[-1]),
    #     # NormalParamExtractor(),
    # ).to(device)
    #
    actor_seq = MLP(
        # out_features=2, num_cells=[64], device=device,
        num_cells=cfg.network.hidden_sizes,
        out_features=action_spec.shape[-1],
        device=device,
    )

    # actor_net_kwargs = {
    #     "num_cells": cfg.network.hidden_sizes,
    #     "out_features": action_spec.shape[-1],
    #     "activation_class": get_activation(cfg),
    # }
    #
    # actor_net = MLP(**actor_net_kwargs)
    #
    actor_module = TensorDictModule(
        # actor_seq, in_keys=["embedding"], out_keys=["loc", "scale"]
        actor_seq,
        in_keys=["embedding"],
        out_keys=["logits"],
    )

    # actor = ProbabilisticActor(
    #     actor_module,
    #     in_keys=["loc", "scale"],
    #     out_keys=["action"],
    #     distribution_class=TanhNormal,
    # )
    actor = ProbabilisticActor(
        spec=CompositeSpec(action=eval_env.action_spec),
        module=actor_module,
        # in_keys=["loc", "scale"],
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # class ValueMLP(nn.Module):
    # def __init__(self, hidden_dim=64):
    # """
    # Initialize the ValueMLP network.
    #
    # Args:
    # input_dim (int): Number of input features. Defaults to 3, i.e., 2+1.
    # hidden_dim (int): Number of neurons in the hidden layers. Defaults to 64.
    # """
    # super(ValueMLP, self).__init__()
    ##self.fc1 = nn.LazyLinear(hidden_dim)
    ##self.relu1 = nn.ReLU()
    # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    # self.relu2 = nn.ReLU()
    ## self.fc3 = nn.Linear(hidden_dim, 1)
    # self.fc3 = nn.Linear(hidden_dim, train_env.action_spec.shape[-1])

    ## def forward(self, x, action):
    # def forward(self, x):
    # """
    # Forward pass of the network.
    #
    # Args:
    # x (torch.Tensor): Input tensor.
    #
    # Returns:
    # torch.Tensor: Output tensor.
    # """
    ## x = torch.cat([x, action], dim=-1)
    ##x = self.relu1(self.fc1(x))
    # x = self.relu2(self.fc2(x))
    # x = self.fc3(x)
    # return x

    # qvalue = ValueOperator(
    # ValueMLP().to(device),
    ## in_keys=["embedding", "action"],
    # in_keys=["embedding"],
    ## out_keys=["state_action_value"],
    # out_keys=["action_value"],
    # )

    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }
    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = TensorDictModule(
        in_keys=["embedding"],
        out_keys=["action_value"],
        module=qvalue_net.to(device),
    )

    # ac_operator = ActorCriticOperator(feature_extractor, actor, qvalue)
    ac_operator = ActorValueOperator(feature_extractor, actor, qvalue)
    # ac_operator = ActorCriticWrapper(actor, qvalue)
    # ac_operator.get_critic_operator()(train_env.reset().to(device))
    ac_operator.get_value_operator()(train_env.reset().to(device))

    # Make policy aware of supplementary inputs and
    # outputs during rollout execution.
    train_env.append_transform(lstm.make_tensordict_primer())
    eval_env.append_transform(lstm.make_tensordict_primer())

    # Combine modules to actor critic model
    model = torch.nn.ModuleList(
        # [ac_operator.get_policy_operator(), ac_operator.get_critic_operator()]
        [ac_operator.get_policy_operator(), ac_operator.get_value_operator()]
    ).to(device)
    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = train_env.reset()
        td = td.to(device)
        # print(td)
        for net in model:
            res = net(td)
            # print(f"After net {res}")
            # print(f"action {res['action']}")
            # if "state_action_value" in res:
            #     print(f"action {res['state_action_value']}")
    del td
    eval_env.close()

    return model


# ====================================================================
# Discrete SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create discrete SAC loss
    loss_module = DiscreteSACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_actions=model[0].spec["action"].space.n,
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        target_entropy_weight=cfg.optim.target_entropy_weight,
        target_entropy=cfg.optim.target_entropy,
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_loss_module_niklas(cfg, model):
    loss = SACLoss(
        # actor_network=model[0].get_policy_operator(),
        # qvalue_network=model[1].get_critic_operator(),
        actor_network=model[0],
        qvalue_network=model[1],
        loss_function="l2",
        target_entropy=0.7,
    )

    # Set update rule for the target network
    updater = SoftUpdate(loss, eps=0.99)

    return loss, updater


def make_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError
