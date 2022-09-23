import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque
from nle import nethack

from torch.nn import functional as F
from netDQN import DQN

# First, we use state just glyphs

class Replaybuffer():
    
    def __init__(self):
        
        self.buffer = deque(maxlen = 10000)
        self.use_cuda = False

    def cache(self, state, next_state, action, reward, done): #나중에 **kwargs로 바꿔보기 feat. JHJ
        
        gly = torch.FloatTensor(state['glyphs'])
        bls = torch.FloatTensor(state['blstats'])
        next_gly = torch.FloatTensor(next_state['glyphs'])
        next_bls = torch.FloatTensor(next_state['blstats'])
        
        self.buffer.append((gly, bls, next_gly, next_bls, torch.LongTensor([action]), torch.FloatTensor([reward]), torch.FloatTensor([done])))
        
    
    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        gly, bls, next_gly, next_bls, action, reward, done = map(torch.stack, zip(*batch)) 
        return gly, bls, next_gly, next_bls, action, reward, done #squeeze? 


    def len(self):
        return len(self.buffer)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    pass
    def __init__(self):
        pass
        # policy network
        self.policy = DQN().to(device)
        # target network
        self.target = DQN().to(device)
        
        # initial optimize
        self.optimizer = torch.optim.Adam(self.policy.parameters())

        self.buffer = Replaybuffer()

        self.gamma = 0.9
        self.batch_size = 2
        self.target_update = 10
    
    def get_action(self, obs):

        # if random.randint() < :
        #     return self.env.action_space.n

        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q = self.policy(observed_glyphs,observed_stats)
        
        _, action = q.max(1) # 가장 좋은 action 뽑기 
        return action.item()

    def update(self):
        gly, bls, next_gly, next_bls, action, reward, done = self.buffer.sample(2)
        
        with torch.no_grad():
            q_next = self.policy(next_gly, next_bls) # batch * action_n
            _, action_next = q_next.max(1) # batch
            q_next_max = self.target(next_gly, next_bls) # batch * action_n
            q_next_max = q_next_max.gather(1, action_next.unsqueeze(1)).squeeze() # #batch
    

        q_target = reward.squeeze() + (1 - done.squeeze()) * self.gamma * q_next_max
        q_curr = self.policy(gly, bls)
        q_curr = q_curr.gather(1, action).squeeze()

        loss = F.smooth_l1_loss(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def train(self):
        episode = 10

        env = gym.make(id = "MiniHack-ETest-v0", observation_keys = ("glyphs","chars","colors","specials","blstats","message"))

        for i in range(episode):
            done = False
            state = env.reset() # each reset generates a new environment instance
            while not done:

                # step
                action = self.get_action(state)
                new_state, reward, done, info =  env.step(action)

                # save buffer 
                self.buffer.cache(state, new_state, action, reward, done)
                # update 
                state = new_state

                if self.buffer.len() > self.batch_size:
                    self.update()

        print("fin")
            # logging 


agent = Agent()
agent.train()

        



        
    


        