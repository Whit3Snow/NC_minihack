import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque


# First, we use state just glyphs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Replaybuffer():
    
    def __init__(self):
        
        self.buffer = deque(maxlen = 10000)
        self.use_cuda = False
        self.num_env = 4

    def cache(self, state, next_state, action, reward, done): #나중에 **kwargs로 바꿔보기 feat. JHJ
        
       
        # breakpoint()

        for i in range(self.num_env):

            gly = torch.FloatTensor(state['glyphs'][i]).to(device)
            bls = torch.FloatTensor(state['blstats'][i]).to(device)
            next_gly = torch.FloatTensor(next_state['glyphs'][i]).to(device)
            next_bls = torch.FloatTensor(next_state['blstats'][i]).to(device)

            # breakpoint()
            self.buffer.append((gly, bls, next_gly, next_bls, torch.LongTensor([action[i]]).to(device), torch.FloatTensor([reward[i]]).to(device), torch.FloatTensor([done[i]]).to(device)))
        
        # breakpoint()
        
    
    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        gly, bls, next_gly, next_bls, action, reward, done = map(torch.stack, zip(*batch)) 
        return gly, bls, next_gly, next_bls, action, reward, done #squeeze? 


    def len(self):
        return len(self.buffer)

    def debuger_sample(self, batch_size):
        pass
        batch = random.sample(self.buffer, batch_size)
        gly, bls, next_gly, next_bls, action, reward, done = map(torch.stack, zip(*batch)) 
        actions = []
        for i in range(1):
            answer = [0,2,1,3,2,0,3,1]
            # random.shuffle(answer)
            actions.extend(answer)

        actions = torch.LongTensor(actions).to("cuda:0")
        # print("debuger action list: \n", actions.unsqueeze(1))

        return gly, bls, next_gly, next_bls, actions.unsqueeze(1), reward, done #squeeze? 
        
