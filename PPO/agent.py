import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque
from nle import nethack

from torch.nn import functional as F
from netPPO import PPO
from memory import Memory

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


# First, we use state just glyphs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, FLAGS = None):

        MOVE_ACTIONS = (nethack.CompassDirection.N,
                    nethack.CompassDirection.E,
                    nethack.CompassDirection.S,
                    nethack.CompassDirection.W,)
                    # nethack.CompassDirection.NE)
                    

        # self.env = gym.vector.make(
        #     id = "MiniHack-Room-5x5-v0",
        #     observation_keys = ("glyphs","blstats"),
        #     actions =  MOVE_ACTIONS,
        #     num_envs = 3)
        self.env = gym.make(
            id = "MiniHack-Room-5x5-v0",
            observation_keys = ("glyphs","blstats"),
            actions =  MOVE_ACTIONS,)

        # self.writer = SummaryWriter()
        
        # if FLAGS.mode == "test":
        #     print("DQN/" + FLAGS.model_dir)
        #     self.policy = torch.load("DQN/" + FLAGS.model_dir)
        #     self.episode = FLAGS.episodes
        #     self.print_freq = FLAGS.print_freq
        #     self.eps_start = FLAGS.eps_start
        #     self.eps_end = FLAGS.eps_end
        #     eps_fraction = 0.3
        #     self.eps_timesteps = eps_fraction * float(self.episode)
        # else:   
        #     # actor network
        #     self.actor = PPO(num_actions= self.env.action_space.n).to(device)
        #     # critic network
        #     self.critic = PPO(num_actions= self.env.action_space.n).to(device)
        
        #     # initial optimize
        #     self.optimizer = torch.optim.Adam(self.policy.parameters())

        #     self.buffer = Replaybuffer()

        #     self.gamma = 0.9
        #     self.batch_size = 32
        #     self.target_update = 50
        #     self.episode = FLAGS.episodes
           
        #     self.print_freq = FLAGS.print_freq
        #     self.save_freq = FLAGS.save_freq
        #     self.model_num = FLAGS.model_num
   

        # # actor network 
        # self.actor = PPO(num_actions= self.env.action_space.nvec[0]).to(device)
        # # critic network
        # self.critic = PPO(num_actions= self.env.action_space.nvec[0]).to(device)
        
        # actor network 
        self.actor = PPO(num_actions= self.env.action_space.n).to(device)
        # critic network
        self.critic = PPO(num_actions= self.env.action_space.n).to(device)

        self.memory = Memory()
 


        self.actor.optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters())

        self.gamma = 0.9
        self.lmbda = 0.7
        self.episode = 3

        # initial optimize
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr = 1e-4)


    def get_action(self, obs):

        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q = self.actor.forward(observed_glyphs,observed_stats)  

        print("sample: ", q.sample())
        print("item: ", q.sample().item())

        return q.sample()



    def calc_advantage(self, gly, bls, next_gly, next_bls, action, reward, done_mask):

        values = self.critic.forward_critic(gly, bls).detach()
        td_target = reward + self.gamma * self.critic.forward_critic(next_gly, next_bls) * done_mask
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0 

        # delta = [[0.02968087] [0.02968087] [0.03968087]] ==> batch_size 

        
        for idx in reversed(range(len(delta))):
            advantage = self.gamma * self.lmbda * advantage + delta[idx][0]
            advantage_lst.append([advantage])
            print(advantage)    
        
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(device)
        return values + advantages, advantages
        #values는 critic 모델
        #return = values + advantages

    def mini_batch(self):
        
        return 

    def update(self):
        batch_size = 3
        clip_param = 0.3
        CRITIC_DISCOUNT = 0.5
        ENTROPY_BETA = 0.001


        gly, bls, next_gly, next_bls, action, reward, log_prob, done_mask, batches = self.memory.sample(batch_size)
        returns, advantages = self.calc_advantage(gly, bls, next_gly, next_bls, action, reward, done_mask)

        PPO_epochs = 3
        for i in range(PPO_epochs):
      
            for batch in batches:       
                gly, bls, action, old_log_probs, return_, advantage = gly[batch], bls[batch], action[batch], log_prob[batch], returns[batch], advantages[batch]
                
                dist = self.actor.forward(gly, bls)
                entropy = dist.entropy().mean()
                new_probs = dist.log_prob(action)

                ratio = (new_probs - old_log_probs).exp()

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.critic.forward_critic(gly, bls).float()
                critic_loss = (return_ - value).pow(2).mean()

                loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy
                
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()




    def train(self):
       
        env = self.env 

        e_rewards = [0.0]
        eps_threshold = 0
        tot_steps = 0
        steps = 0

        n_steps = 0
        N = 20

        for epi in range(self.episode):
            done = False
            state = env.reset() # each reset generates a new environment instance    
            
            while not done:
                pass
                n_steps += 1
                action = self.get_action(state)
                new_state, reward, done, info =  env.step(action.item())

                observed_glyphs = torch.from_numpy(state['glyphs']).float().unsqueeze(0).to(device)
                observed_stats = torch.from_numpy(state['blstats']).float().unsqueeze(0).to(device)

                dist = self.actor.forward(observed_glyphs, observed_stats)
                log_prob = dist.log_prob(action).item()
                self.memory.cache(state, new_state, action, reward, log_prob, done)
                
                if n_steps % N == 0:
                    print("in there: ", n_steps)
                    self.update()

                state = new_state
            
            print("done", "@" * 40)

            

    def test(self):
        pass
        
        env = self.env 
        state = env.reset()
        #state 형태: key:[[], ... ,[]], key2:[[], ... ,[]], ...


        action = self.get_action(state)
        for i in range(3):
            new_state, reward, done, info =  env.step(action)

            self.memory.cache(state, new_state, action, reward, done)
        
        self.memory.sample(3)

        self.update()


agent = Agent()
agent.train()
    


        