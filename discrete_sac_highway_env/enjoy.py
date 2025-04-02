# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Discrete SAC Example.

This is a simple self-contained example of a discrete SAC training script.

It supports gym state environments like CartPole.

The helper functions are coded in the utils.py associated with this script.
"""

import time

import hydra
import numpy as np
import torch
import torch.cuda

from torchrl.envs.utils import ExplorationType, set_exploration_type

from utils import (
    make_environment,
    make_sac_agent,
    make_sac_agent_lstm,
)

import tqdm


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=None)

    eval_env.auto_register_info_dict()

    # Create agent
    # model = make_sac_agent(cfg, train_env, eval_env, device)
    if cfg.network.use_lstm:
        model = make_sac_agent_lstm(cfg, train_env, eval_env, device)
    else:
        model = make_sac_agent(cfg, train_env, eval_env, device)

    # load saved model
    model[0].load_state_dict(torch.load(cfg.eval.load_path, map_location=device))

    pbar = tqdm.tqdm(total=cfg.eval.num_steps)

    crashes = 0
    merges = 0
    episodes = 0
    last_info = None
    speed = 0
    road_speed = 0

    def step_callback(env, td):
        nonlocal last_info
        nonlocal crashes
        nonlocal episodes
        nonlocal merges
        nonlocal pbar
        nonlocal speed
        nonlocal road_speed

        # env.render()
        if td["next"]["done"]:
            episodes += 1

            if td["next"]["crashed"]:
                crashes += 1
            if td["next"]["merged"]:
                merges += 1

        speed += td["next"]["average_speed"]
        road_speed += td["next"]["average_road_speed"]

        pbar.update(1)
        # time.sleep(0.03)

    eval_rollout_steps = cfg.eval.num_steps
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        eval_rollout = eval_env.rollout(
            eval_rollout_steps,
            model[0],
            auto_cast_to_device=True,
            break_when_any_done=False,
            callback=step_callback,
        )

    print(f"Crashrate {crashes/episodes}")
    print(f"Mergerate {merges/episodes}")
    print(f"Average ego vehicle speed {speed/cfg.eval.num_steps}")
    print(f"Average speed of all cars {road_speed/cfg.eval.num_steps}")


if __name__ == "__main__":
    main()
