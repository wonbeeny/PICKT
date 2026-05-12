"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import math
import numpy as np
# from omegaconf.dictconfig import DictConfig
from torch.optim.lr_scheduler import LambdaLR


# def cfg_to_hparams(cfg, hparam_dict, parent_str=""):
#     for key, val in cfg.items():
#         if isinstance(val, DictConfig):
#             hparam_dict = cfg_to_hparams(val, hparam_dict, parent_str + key + "__")
#         else:
#             hparam_dict[parent_str + key] = str(val)
#     return hparam_dict


def linear_scheduler(optimizer, warmup_steps, training_steps, last_epoch=-1):
    """linear_scheduler with warmup from huggingface"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            _lr = float(current_step) / float(max(1, warmup_steps))
            return _lr
        _lr = max(
            0.0,
            float(training_steps - current_step)
            / float(max(1, training_steps - warmup_steps)),
        )
        return _lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_scheduler(
    optimizer, warmup_steps, training_steps, cycles=0.5, last_epoch=-1
):
    """Cosine LR scheduler with warmup from huggingface"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            _lr = current_step / max(1, warmup_steps)
            return _lr
        progress = current_step - warmup_steps
        progress /= max(1, training_steps - warmup_steps)
        _lr  = max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2 * progress)))
        return _lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def multistep_scheduler(optimizer, warmup_steps, milestones, gamma=0.1, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # calculate a warmup ratio
            _lr = current_step / max(1, warmup_steps)
            return _lr
        else:
            # calculate a multistep lr scaling ratio
            idx = np.searchsorted(milestones, current_step)
            _lr = gamma ** idx
            return _lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)