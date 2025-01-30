"""Enjoy the trained model.
Loads a model checkpoint and evaluates it on the given environment
"""

import time

import hydra
import numpy as np
import torch
import torch.cuda

from utils import (
    make_environment,
    make_sac_agent_new,
)


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
    train_env, eval_env = make_environment(cfg)

    # Create agent
    model = make_sac_agent_new(cfg, train_env, eval_env, device)
    model[0].load_state_dict(torch.load(cfg.eval.checkpoint, map_location=device))

    td = eval_env.reset()
    for _ in range(100):
        action = model[0](td)
        td = eval_env.step(action)
        eval_env.env.render()

        if td[("next", "done")]:
            td = eval_env.reset()

        time.sleep(0.05)


if __name__ == "__main__":
    main()
