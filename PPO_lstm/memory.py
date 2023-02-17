import torch
import minihack 
from torch import nn
import numpy as np
from collections import deque


# First, we use state just glyphs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory():
    
    def __init__(self, p_batch_size):
        
        self.buffer = deque(maxlen = p_batch_size)

    def cache(self, state, next_state, action, reward, log_prob, done, h_t, c_t, next_h_t, next_c_t): #나중에 **kwargs로 바꿔보기 feat. JHJ
        
        gly = torch.FloatTensor(state['glyphs']).to(device)
        bls = torch.FloatTensor(state['blstats']).to(device)
        next_gly = torch.FloatTensor(next_state['glyphs']).to(device)
        next_bls = torch.FloatTensor(next_state['blstats']).to(device)
        

        self.buffer.append((gly, bls, next_gly, next_bls, torch.LongTensor(action.tolist()).to(device), torch.FloatTensor(reward).to(device), torch.FloatTensor(log_prob).to(device), torch.FloatTensor(done).to(device),h_t, c_t, next_h_t, next_c_t))
        
    
    def sample(self):

        gly, bls, next_gly, next_bls, action, reward, log_prob, dones, h_ts, c_ts, next_h_ts, next_c_ts = map(torch.stack, zip(*self.buffer)) 

        done_lst = []

        
        def versa(a):
            return 1 - a

        done_lst = list(map(versa, dones))
        done_lst = torch.stack(done_lst, dim = 0)

        return gly, bls, next_gly, next_bls, action, reward, log_prob, done_lst, h_ts, c_ts, next_h_ts, next_c_ts #squeeze? 
    

    def len(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


