# -*- coding: utf-8 -*-

import os
import random

# @Time  : 2023/1/24 17:01
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import numpy as np
import torch


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
