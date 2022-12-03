# -*-  coding = utf-8 -*-
# @Time : 2022/10/9 10:25 下午
# @author : Wang Zhixian
# @File : setalpha.py
# @Software: PyCharm

import numpy as np
import torch

def target_alpha(targets):
    target = targets.numpy()  # change it into the ndarray

    def gen_onehot(category, total_cat=2):
        label = np.ones(total_cat)
        label[category] = 20
        return label

    target_alphas = []
    for i in target:
        if i == 10:
            target_alphas.append(np.ones(2))
        else:
            target_alphas.append(gen_onehot(i))
    return torch.Tensor(target_alphas)