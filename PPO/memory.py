import torch
import minihack 
from torch import nn
import numpy as np
from collections import deque


# First, we use state just glyphs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory():
    
    def __init__(self, p_batch_size, buffer_size):
        
        self.buffer = deque(maxlen = p_batch_size * buffer_size)
        self.use_cuda = False

    def cache(self, state, next_state, action, reward, log_prob, done): #나중에 **kwargs로 바꿔보기 feat. JHJ
        
        gly = torch.FloatTensor(state['glyphs']).to(device)
        bls = torch.FloatTensor(state['blstats']).to(device)
        next_gly = torch.FloatTensor(next_state['glyphs']).to(device)
        next_bls = torch.FloatTensor(next_state['blstats']).to(device)
        
        self.buffer.append((gly, bls, next_gly, next_bls, torch.LongTensor([action]).to(device), torch.FloatTensor([reward]).to(device), torch.FloatTensor([log_prob]).to(device), torch.FloatTensor([done]).to(device)))
        
    
    def sample(self):

        gly, bls, next_gly, next_bls, action, reward, log_prob, dones = map(torch.stack, zip(*self.buffer)) 

        done_lst = []
        for i in range(len(dones)):
            done = 0.0 if dones[i] else 1.0
            done_lst.append([done])

        done_lst = torch.tensor(done_lst, dtype = torch.float).to(device)


        return gly, bls, next_gly, next_bls, action, reward, log_prob, done_lst #squeeze? 
    

    def len(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


