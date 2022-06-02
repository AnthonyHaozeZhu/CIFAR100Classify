# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：utils.py
@Author ：AnthonyZ
@Date ：2022/6/2 16:28
"""

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

