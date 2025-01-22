import torch
from tensordict import TensorDict
from torchrl.objectives.value.functional import (
    _inv_pad_sequence,
    _split_and_pad_sequence,
)
from torchrl.objectives.value.utils import _get_num_per_traj_init

is_init = torch.zeros(4, 5, dtype=torch.bool)
is_init[:, 0] = True
is_init[0, 3] = True
is_init[1, 2] = True

tensordict = TensorDict(
    {
        "is_init": is_init,
        "obs": torch.arange(20).view(4, 5).unsqueeze(-1).expand(4, 5, 3),
        "mask": torch.ones(4, 5),  # .unsqueeze(-1).expand(4, 5, 3),
    },
    [4, 5],
)
print("original")
print(tensordict)
print(tensordict["obs"])
splits = _get_num_per_traj_init(is_init)
# print(splits)
td = _split_and_pad_sequence(tensordict, splits)
print("non inverted")
print(td)
# print(td["is_init"])
# print(td["mask"])

td_inv = _inv_pad_sequence(td, splits)
print("inverted")
print(td_inv)
# print(td_inv["mask"])
print(td_inv["obs"])
