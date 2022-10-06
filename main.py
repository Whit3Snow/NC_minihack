import argparse 
import torch
from torch import nn
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
import importlib

import gym


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default = '10000', type = int, help = '총 step수를 일컬음')
    parser.add_argument('--mode', default = "test", type = str)
    parser.add_argument('--agent', default = "DQN", type = str)
    parser.add_argument('--eps-start', default = 1.0, type = float, help = "e-greedy 시작 값")
    parser.add_argument('--eps-end', default = 0.1, type = float, help = "e-greedy 마지막 값")
    parser.add_argument('--print_freq', default = 25, type = int, help = "log 주기")
    parser.add_argument('--save_freq', default = 100, type = int, help = "mode_save 주기")
    parser.add_argument("--model_dir", type = str, help = "model 확인")

    global FLAGS
    FLAGS = parser.parse_args()
    

    module = FLAGS.agent + ".agent" 
    name = "Agent"
    agent = getattr(importlib.import_module(module), name)(FLAGS) # import 하고 싶은 파일

    if FLAGS.mode == 'train':
        # assert 1 == 2
        agent.train()
    elif FLAGS.mode == 'test':
        agent.test()


if __name__ == "__main__":
    main()