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
            # max_episode_steps= 1000,)

        self.writer = SummaryWriter()

        self.batch_size = 32
        self.buffer_size = 1
        
        #test
        # if test == True:
        #     self.actor = torch.load("distance_1/" + "model")


        # actor network 
        self.actor = PPO(num_actions= self.env.action_space.n).to(device)
        # critic network
        self.critic = PPO(num_actions= self.env.action_space.n).to(device)

        self.memory = Memory(self.batch_size, self.buffer_size)
        self.print_freq = self.batch_size * self.buffer_size  #640
        self.save_freq = self.batch_size * self.buffer_size * 2


        self.gamma = 0.99
        self.lmbda = 0.95
        self.episode = 10000

        # initial optimize
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 1e-4)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 0.001)


    def get_action(self, obs):

        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q = self.actor.forward(observed_glyphs,observed_stats, True)  

        return q.sample(), q



    def calc_advantage(self, gly, bls, next_gly, next_bls, action, reward, done_mask):
        

        values = self.critic.forward(gly, bls, False).detach()
        td_target = reward + self.gamma * self.critic.forward(next_gly, next_bls, False) * done_mask
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
        clip_param = 0.2
        CRITIC_DISCOUNT = 0.5
        ENTROPY_BETA = 0.01


        gly, bls, next_gly, next_bls, action, reward, log_prob, done_mask = self.memory.sample()
        returns, advantages = self.calc_advantage(gly, bls, next_gly, next_bls, action, reward, done_mask)# batch_size
 
        PPO_epochs = 10
        n = self.buffer_size * self.batch_size
        arr = np.arange(n)
        for j in range(PPO_epochs):
            np.random.shuffle(arr)
            for i in range(self.buffer_size):
                batch_index = arr[self.batch_size * i: self.batch_size * (i + 1)]       
                batch = torch.LongTensor(batch_index)

                gly_, bls_, action_, old_log_probs, return_, advantage = gly[batch], bls[batch], action[batch], log_prob[batch], returns[batch], advantages[batch]
                
                dist = self.actor.forward(gly_, bls_, True)
                entropy = dist.entropy().mean()
                new_probs = dist.log_prob(action_)
                ratio = (new_probs - old_log_probs).exp()

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.critic.forward(gly_, bls_, False).float()
                critic_loss = (return_ - value).pow(2).mean()

                loss = (CRITIC_DISCOUNT * critic_loss) + actor_loss  # - (ENTROPY_BETA * entropy)
                
                
                self.actor_optimizer.zero_grad()
                # self.critic_optimizer.zero_grad()
                
                loss.backward()
                self.actor_optimizer.step()
                # self.critic_optimizer.step()

        self.memory.clear()

        return loss 



    def train(self):
       
        env = self.env 
        e_rewards = [0.0]
        tot_steps = 0
        n_steps = 0
        steps = 0
        loss = []
        action_steps = []

        save = 0
        max_step = 40
        state = env.reset() # each reset generates a new environment instance    

        while True:
            state = env.reset()
            # env.render("human")
            for i in range(max_step):

                steps += 1
                tot_steps += 1
                n_steps += 1

                action, dist = self.get_action(state)

                new_state, reward, done, info =  env.step(action.item())
                
                # action_arr.append(action.item())
                e_rewards[-1] += reward
                
                log_prob = dist.log_prob(action).item()
                self.memory.cache(state, new_state, action, reward, log_prob, done)
                
                if tot_steps % (self.buffer_size * self.batch_size) == 0:
                    loss.append(self.update())

                state = new_state

                
                if done:
                    break
            
            action_steps.append(n_steps)
            n_steps = 0
            e_rewards.append(0.0)

            if len(e_rewards) % self.print_freq == 0 :
                save += 1
                print("************************************************")
                print("mean_steps: {} and tot_steps: {}".format(sum(action_steps)/ len(action_steps), tot_steps))
                print("num_episodes: {} {}".format(len(e_rewards), len(action_steps)))
                print("mean 100 episode reward: {}".format(round(np.mean(e_rewards), 2)))
                print("************************************************")
                
                if save % 50 == 0:
                    torch.save(self.actor, "{}/model".format("test5") + str(save))

                self.writer.add_scalar("mean_reward", round(np.mean(e_rewards[-101:-1]), 2), len(e_rewards) / (self.buffer_size * self.batch_size))
                self.writer.add_scalar("mean_steps", steps / (self.buffer_size * self.batch_size), len(e_rewards) / (self.buffer_size * self.batch_size))
                self.writer.add_scalar("mean_loss", sum(loss)/ len(loss), len(e_rewards) /(self.buffer_size * self.batch_size))

                steps = 0
                loss = []
                e_rewards = [0.0]
                action_steps = []
                
            #temp#


            # if freq % self.save_freq == 0:
            #     print("save the model!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #     torch.save(self.actor, "{}/model".format(self.model_num) + str(tot_steps))
  





    np.set_printoptions(threshold=np.inf, linewidth=np.inf) #for debuger
    def test(self):
  
        actions = []
        env = self.env
        e_rewards = [0.0]
        tot_steps = 0
        steps = 0

        state = env.reset() # each reset generates a new environment instance       

        for epi in range(100):
            done = False
            state = env.reset() # each reset generates a new environment instance       
            env.render("human")


            while not done:
                steps += 1
                tot_steps += 1
                # step
                action = self.get_action(state)
                actions.append(action.item())
                new_state, reward, done, info =  env.step(action.item())
                state = new_state

                e_rewards[-1] += reward

            
            print(actions)
            breakpoint()
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
# agent.test()

    


        