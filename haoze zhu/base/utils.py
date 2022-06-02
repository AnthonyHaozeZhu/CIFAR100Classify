# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：utils.py
@Author ：AnthonyZ
@Date ：2022/6/2 16:28
"""

import matplotlib.pyplot as plt
import numpy as np
import importlib


def create_dataloader(opt):
    dataloader = importlib.import_module("dataset." + opt.dataname)
    return dataloader.ImageTextDataloader(opt)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

