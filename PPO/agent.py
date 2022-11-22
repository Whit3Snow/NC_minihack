import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque
from nle import nethack

from torch.nn import functional as F
from PPO.netPPO import PPO
from PPO.memory import Memory

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

        self.batch_size = 20
        self.buffer_size = 1
        self.flags = FLAGS
        
        if FLAGS.mode == "test":
            self.actor_critic = torch.load("PPO/" + FLAGS.model_dir)

            self.env = gym.make(
                    id = "MiniHack-Room-5x5-v0",
                    observation_keys = ("glyphs","blstats"),
                    actions =  MOVE_ACTIONS,)
        else: 
            # actor network 
            self.actor_critic = PPO(num_actions= self.env.action_space.n).to(device)
            # critic network
            # self.critic = PPO(num_actions= self.env.action_space.n).to(device)

            self.memory = Memory(self.batch_size, self.buffer_size)
            self.print_freq = self.batch_size * self.buffer_size  

            self.gamma = 0.99
            self.lmbda = 0.95
            self.episode = 10000
            self.model_num = FLAGS.model_num

            # initial optimize
            self.actor_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = 1e-4)
            # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 0.001)
            self.num_envs = FLAGS.num_actors
            
            self.env = gym.vector.make(
                id = "MiniHack-Room-5x5-v0",
                observation_keys = ("glyphs","blstats"),
                actions =  MOVE_ACTIONS,
                num_envs =  self.num_envs, # self.num_envs,
                max_episode_steps= 20,

                )



    def get_action(self, obs):

        if self.flags.mode == "test":
            observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
            observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)
        else:
            observed_glyphs = torch.from_numpy(obs['glyphs']).float().to(device)
            observed_stats = torch.from_numpy(obs['blstats']).float().to(device)

        with torch.no_grad():
            q = self.actor_critic.forward(observed_glyphs,observed_stats, True)  

        return q.sample(), q

    def calc_advantage2(self, gly, bls, next_gly, next_bls, action, reward, done_mask):

        batch_size = done_mask.shape[0]
        value_old_state = self.actor_critic.forward(gly, bls, False).squeeze(1).detach()
        value_new_state = self.actor_critic.forward(next_gly, next_bls, False).squeeze(1).detach()
        advantage = np.zeros(batch_size + 1)

        for t in reversed(range(batch_size)):
            delta = reward[t] + (self.gamma * value_new_state[t] * done_mask[t]) - value_old_state[t]
            advantage[t] = delta + (self.gamma * self.lmbda * advantage[t + 1] * done_mask[t])

        advantage = torch.tensor(advantage[:batch_size], dtype=torch.float).to(device)
        value_target = advantage[:batch_size] + np.squeeze(value_old_state)

        return value_target, advantage[:batch_size]



    def calc_advantage(self, gly, bls, next_gly, next_bls, action, reward, done_mask):

        
        values = self.actor_critic.forward(gly, bls, False).squeeze(1).detach()
        td_target = reward + self.gamma * self.actor_critic.forward(next_gly, next_bls, False).squeeze(1) * done_mask
        delta = td_target - values
        
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0 

        for idx in reversed(range(len(delta))):
            advantage = (self.gamma * self.lmbda * advantage) + delta[idx]
            advantage_lst.append([advantage])                                       

        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(device)
        returns = values + advantages.squeeze(1)

        return returns, advantages
        #values는 critic 모델
        #return = values + advantages


    def update(self):
        clip_param = 0.1
        CRITIC_DISCOUNT = 0.5
        ENTROPY_BETA = 0.01


        gly, bls, next_gly, next_bls, action, reward, log_prob, done_mask = self.memory.sample()

        # breakpoint()

        gly = gly.view([-1, 21, 79])
        bls = bls.view([-1, 26])
        next_gly = next_gly.view([-1, 21, 79])
        next_bls = next_bls.view([-1, 26])
        reward = reward.reshape(self.flags.num_actors * self.batch_size * self.buffer_size)
        done_mask = done_mask.reshape(self.flags.num_actors * self.batch_size * self.buffer_size)
        log_prob = log_prob.reshape(self.flags.num_actors * self.batch_size * self.buffer_size)
        action = action.reshape(self.flags.num_actors * self.batch_size * self.buffer_size)

        returns, advantages = self.calc_advantage2(gly, bls, next_gly, next_bls, action, reward, done_mask)# batch_size
        # breakpoint()
    
        PPO_epochs = 10
        n = self.buffer_size * self.batch_size
        arr = np.arange(n)
        for j in range(PPO_epochs):
            np.random.shuffle(arr)
            for i in range(self.buffer_size):
                batch_index = arr[self.batch_size * i: self.batch_size * (i + 1)]       
                batch = torch.LongTensor(batch_index)

                gly_, bls_, action_, old_log_probs, return_, advantage = gly[batch], bls[batch], action[batch], log_prob[batch], returns[batch], advantages[batch]
                
                dist = self.actor_critic.forward(gly_, bls_, True)
                entropy = dist.entropy().mean()
                new_probs = dist.log_prob(action_)
                ratio = (new_probs - old_log_probs).exp()

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()

                # breakpoint()

                value = self.actor_critic.forward(gly_, bls_, False).float()
                critic_loss = (return_ - value).pow(2).mean() # MSE_loss

                loss = (CRITIC_DISCOUNT * critic_loss) + actor_loss #- (ENTROPY_BETA * entropy)
                
                # for name,params in self.actor_critic.named_parameters():
                #     print("name:", name)
                #     print(params)
                #     print("=========" * 8)
                # breakpoint()

                self.actor_optimizer.zero_grad()
                # self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
                self.actor_optimizer.step()

                # for name,params in self.critic.named_parameters():
                #     print("name:", name)
                #     print(params)
                #     print("=========" * 8)
                # breakpoint()

                # self.critic_optimizer.step()
            

        self.memory.clear()

        return loss 



    def train(self):
       
        env = self.env 
        e_rewards = [0.0] * self.num_envs
        tot_steps = 0
        steps = [0 for i in range(self.num_envs)]
        step = [1 for i in range(self.num_envs)]

        done_steps = []
        done_rewards = []
        
        n_steps = 0
        step_s = 0
        loss = []
        action_steps = []

        save = 0
        print_freq = 1
        max_step = 20
        state = env.reset() # each reset generates a new environment instance    

        while True:
            
            for mini_step in range(max_step):

                step_s += 1
                tot_steps += 1
                n_steps += 1

                action, dist = self.get_action(state) # action : tensor([2, 2, 0, 0], device='cuda:0') ,  dist : Categorical(probs: torch.Size([4, 4]))

                new_state, reward, done, info =  env.step(action.tolist())
        
                e_rewards = [x + y for x,y in zip(e_rewards, reward)] # num_envs

                log_prob = dist.log_prob(action).tolist()
                self.memory.cache(state, new_state, action, reward, log_prob, done)
                
                if mini_step == max_step - 1:
                    loss.append(self.update())

                state = new_state

                steps = [x + y for x,y in zip(steps, step)] # num_envs
                done_idx = [i for i,ele in enumerate(done) if ele == True]

                for k in done_idx:
                    done_steps.append(steps[k]) #envs에서 done된 것만 step 가져오기
                    steps[k] = 0 
                    done_rewards.append(e_rewards[k])
                    e_rewards[k] = 0.0 

            
            action_steps.append(n_steps)
            n_steps = 0

            # print("n_steps: ",n_steps)
            # print("len(done_rewards): ", len(done_rewards))
            # e_rewards.append(0.0)

            if len(done_rewards) // 100 > print_freq :
                save += 1
                print_freq += 1
                print("************************************************")
                print("mean_steps: {} and tot_steps: {}".format(np.mean(done_steps[-101:-1]), tot_steps))
                print("num_episodes: {} {}".format(len(done_rewards), len(action_steps)))
                print("mean 100 episode reward: {}".format(round(np.mean(done_rewards[-101:-1]), 2)))
                print("************************************************")
                
                if save % 50 == 0:
                    torch.save(self.actor_critic, "PPO/{}/model".format(self.model_num) + str(save))

                # self.writer.add_scalar("mean_reward", round(np.mean(e_rewards[-101:-1]), 2), len(e_rewards) / (self.buffer_size * self.batch_size))
                # self.writer.add_scalar("mean_steps", step_s / (self.buffer_size * self.batch_size), len(e_rewards) / (self.buffer_size * self.batch_size))
                # self.writer.add_scalar("mean_loss", sum(loss)/ len(loss), len(e_rewards) /(self.buffer_size * self.batch_size))

                step_s = 0
                loss = []
                # e_rewards = [0.0] * self.num_envs
                action_steps = []
                



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
                action, dist = self.get_action(state)
                actions.append(action.item())
                new_state, reward, done, info =  env.step(action.item())
                state = new_state

                e_rewards[-1] += reward

            
            print(actions, e_rewards[-1], len(actions))
            breakpoint()
            actions = []

            # 한번 episode 시행 후------------------------------------------------------------------------------------
            e_rewards.append(0.0)
        
            #logging
            if len(e_rewards) % 20 == 0 :

                print("************************************************")
                print("means_steps: {} and tot_steps: {}".format(steps/25, tot_steps))
                print("num_episodes: {}".format(len(e_rewards)))
                print("mean 100 episode reward: {}".format(round(np.mean(e_rewards[-101:-1]), 2)))
                print("************************************************")
                steps = 0

        