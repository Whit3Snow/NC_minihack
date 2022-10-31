import argparse 
import torch
from torch import nn
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
import importlib
import os
import gym

def createFolder(directory):
    try: 
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)
    

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default = '50000', type = int, help = '총 step수를 일컬음')
    parser.add_argument('--mode', default = "test", type = str)
    parser.add_argument('--agent', default = "DQN", type = str)
    parser.add_argument('--eps-start', default = 0.9, type = float, help = "e-greedy 시작 값")
    parser.add_argument('--eps-end', default = 0.01, type = float, help = "e-greedy 마지막 값")
    parser.add_argument('--print_freq', default = 27, type = int, help = "log 주기")
    parser.add_argument('--save_freq', default = 100, type = int, help = "mode_save 주기")
    parser.add_argument("--model_dir", type = str, help = "model 확인")
    parser.add_argument("--model_num", default = "model", type = str, help = "model 디렉토리 숫자")
    parser.add_argument("--lr", default = 1e-7, type = float, help = "just learning rate")

    parser.add_argument("--env", default = "MiniHack-Room-Random_curi-5x5-v0", type = str, help = "just gym environment by JHJ")
    '''
    # Goal env list 
        0. MiniHack-Room-5x5-v0
        1. MiniHack-Corridor-R2-v0
        MiniHack-Room-Random_curi-5x5-v0

    '''
    

    global FLAGS
    FLAGS = parser.parse_args()
    

    module = FLAGS.agent + ".agent" 
    name = "Agent"
    agent = getattr(importlib.import_module(module), name)(FLAGS) # import 하고 싶은 파일

    if FLAGS.mode == 'train':
        createFolder("./" + FLAGS.agent + "/" + FLAGS.model_num)
        agent.train()
    elif FLAGS.mode == 'test':
        agent.test()


if __name__ == "__main__":
    main()