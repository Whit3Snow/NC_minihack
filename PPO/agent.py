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
            id = "MiniHack-Room-Random_curi-5x5-v0",
            observation_keys = ("glyphs","blstats"),
            actions =  MOVE_ACTIONS,
            max_episode_steps= 1000,)

        self.writer = SummaryWriter()

        
        # actor network 
        self.actor = PPO(num_actions= self.env.action_space.n).to(device)
        # critic network
        self.critic = PPO(num_actions= self.env.action_space.n).to(device)

        self.memory = Memory()
        self.print_freq = 25
        self.save_freq = 100

        test = False
        if test:
            self.actor = torch.load("/" + FLAGS.model_dir)
        else:
            self.actor.optimizer = torch.optim.Adam(self.actor.parameters())
            self.critic.optimizer = torch.optim.Adam(self.critic.parameters())

        self.gamma = 0.99
        self.lmbda = 0.95
        self.episode = 10000

        # initial optimize
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr = 1e-4)


    def get_action(self, obs):

        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q = self.actor.forward(observed_glyphs,observed_stats)  

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
    
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(device)
        return values + advantages, advantages
        #values는 critic 모델
        #return = values + advantages


    def update(self):
        batch_size = 32
        clip_param = 0.2
        CRITIC_DISCOUNT = 0.5
        ENTROPY_BETA = 0.001


        gly, bls, next_gly, next_bls, action, reward, log_prob, done_mask, batches = self.memory.sample(batch_size)
        returns, advantages = self.calc_advantage(gly, bls, next_gly, next_bls, action, reward, done_mask)

        PPO_epochs = 3

        # print("action:", action)
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

        return loss 



    def train(self):
       
        env = self.env 
        e_rewards = [0.0]
        tot_steps = 0
        steps = 0
        n_steps = 0
        N = 32
        loss = 0

        
        while True:
            state = env.reset(size = 5, distance = 1) # each reset generates a new environment instance    
            # env.render("human")
            print(state["glyphs"])
            for epi in range(self.episode)
                
                steps += 1
                tot_steps += 1

                n_steps += 1
                action = self.get_action(state)
                new_state, reward, done, info =  env.step(action.item())

                e_rewards[-1] += reward

                observed_glyphs = torch.from_numpy(state['glyphs']).float().unsqueeze(0).to(device)
                observed_stats = torch.from_numpy(state['blstats']).float().unsqueeze(0).to(device)

                dist = self.actor.forward(observed_glyphs, observed_stats)
                log_prob = dist.log_prob(action).item()
                self.memory.cache(state, new_state, action, reward, log_prob, done)
                
                if n_steps % N == 0:
                    loss += self.update()

                state = new_state
            
            e_rewards.append(0.0)
            # print("Episode: ", epi, "  / step: ", tot_steps )
            
            if len(e_rewards) % self.print_freq == 0 :
                print("************************************************")
                print("mean_steps: {} and tot_steps: {}".format(steps / 25, tot_steps))
                print("num_episodes: {}".format(len(e_rewards)))
                print("mean 100 episode reward: {}".format(round(np.mean(e_rewards[-101:-1]), 2)))
                print("************************************************")

                self.writer.add_scalar("mean_reward", round(np.mean(e_rewards[-101:-1]), 2), len(e_rewards) / 25)
                self.writer.add_scalar("mean_steps", steps / 25, len(e_rewards) / 25)
                self.writer.add_scalar("mean_loss", loss / 25, len(e_rewards) / 25)
                steps = 0
                loss = 0
            

            #temp#
            self.model_num = "distance_1"


            if len(e_rewards) % self.save_freq == 0:
                torch.save(self.actor, "{}/model".format(self.model_num) + str(len(e_rewards)))
  


            

    def test(self):


        actions = []
        env = self.env
        e_rewards = [0.0]
        tot_steps = 0
        steps = 0


        for epi in range(self.episode):
            done = False
            state = env.reset() # each reset generates a new environment instance
            # steps= 0        
            
            while not done:
                steps += 1
                tot_steps += 1
                # step
                action = self.get_action(state)
                
                actions.append(action)
                
                new_state, reward, done, info =  env.step(action)
                state = new_state

                e_rewards[-1] += reward

                # print("Episode: ", epi, "  / step: ", tot_steps, "\tAction Taken: ", str(action) )
                # env.render("human")
                # if action != 1:
                #     print("action: ", action)
            
            actions = []

            # 한번 episode 시행 후------------------------------------------------------------------------------------
            e_rewards.append(0.0)
     
            #logging
            if len(e_rewards) % self.print_freq == 0 :

                print("************************************************")
                print("means_steps: {} and tot_steps: {}".format(steps/25, tot_steps))
                print("num_episodes: {}".format(len(e_rewards)))
                print("mean 100 episode reward: {}".format(round(np.mean(e_rewards[-101:-1]), 2)))
                print("************************************************")
                steps = 0



agent = Agent()
agent.train()
    


        