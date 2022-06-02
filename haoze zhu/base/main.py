# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：main.py.py
@Author ：AnthonyZ
@Date ：2022/6/2 14:58
"""


import argparse
from data import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data", type=str, help="The input data dir")
    parser.add_argument("--batch_size", default=32, type=int, help="The batch size of training")
    parser.add_argument("--device", default='cuda', type=str, help="The training device")

    args = parser.parse_args()

    train_loader, test_loader = cifar100_dataset(args)
