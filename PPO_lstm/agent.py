import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque
from nle import nethack

from torch.nn import functional as F
from PPO_lstm.netPPO import Policy, Value
from PPO_lstm.memory import Memory

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

import torch.nn.functional as F



# First, we use state just glyphs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

class Agent():
    
    def __init__(self, FLAGS = None):


        MOVE_ACTIONS = (nethack.CompassDirection.N,
                    nethack.CompassDirection.E,
                    nethack.CompassDirection.S,
                    nethack.CompassDirection.W,)


        if FLAGS.debug == 0: #debugging이 아닌 경우, 즉 tensorboard에 기록할 경우
            self.writer = SummaryWriter('runs/' + FLAGS.model_num)

        

        if FLAGS.mode == "test":
            self.flags = FLAGS

            self.policy = torch.load(FLAGS.model_dir)

            self.env = gym.make(
                    id = "Minihack-CurriRoom-v0",
                    # id = "MiniHack-Corridor-R2-v0",
                    # id = "MiniHack-Room-5x5-v0",

                    observation_keys = ("glyphs","blstats"),
                    actions =  MOVE_ACTIONS,)
            
            self.h_t_policy = torch.zeros(1, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.
            self.c_t_policy = torch.zeros(1, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.

          

        else: 
                        
            self.flags = FLAGS
            embed_dim = 16
            n_layers = 3
            self.num_envs = FLAGS.num_actors
            self.batch_size = 150
            self.gamma = 0.99 # value 값을 줄이면, 
            self.lmbda = 0.95
            self.model_num = FLAGS.model_num
            self.print_freq = self.batch_size
            
            self.env = gym.vector.make(
                id = "Minihack-CurriRoom-v0",
                # id = "MiniHack-Room-5x5-v0",
                # id = "MiniHack-Corridor-R2-v0",
                # id = "MiniHack-MultiRoom-N4-v0",
                observation_keys = ("glyphs","blstats"),
                actions =  MOVE_ACTIONS,
                num_envs =  self.num_envs,
                max_episode_steps= 150,)

            # actor_critic network 
            self.policy = Policy(num_actions= self.env.single_action_space.n, embedding_dim= embed_dim, num_layers= n_layers).to(device)
            self.value = Value(num_actions= self.env.single_action_space.n, embedding_dim= embed_dim, num_layers= n_layers).to(device)
            self.memory = Memory(self.batch_size)
            self.h_t_policy = torch.zeros(self.num_envs, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.
            self.c_t_policy = torch.zeros(self.num_envs, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.

            self.h_t_value = torch.zeros(self.num_envs, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.
            self.c_t_value = torch.zeros(self.num_envs, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.
            # initial optimize
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = FLAGS.lr)
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr = FLAGS.lr)

           
    
            
    def get_action(self, obs):

        if self.flags.mode == 'test':
            observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
            observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)


        else:
            observed_glyphs = torch.from_numpy(obs['glyphs']).float().to(device)
            observed_stats = torch.from_numpy(obs['blstats']).float().to(device)

        with torch.no_grad():
            dist, self.h_t_policy, self.c_t_policy = self.policy.forward_lstm(observed_glyphs,observed_stats, self.h_t_policy, self.c_t_policy)  



        return dist.sample(), dist

  

    def calc_advantage(self, gly, bls, next_gly, next_bls, reward, done_mask, h_ts, c_ts, next_h_ts, next_c_ts):
        
        batch_size = done_mask.shape[0]
        value_old_state = self.value.forward_lstm(gly, bls, h_ts, c_ts).squeeze(1).detach() # [600, 1] --> [600]
        value_new_state = self.value.forward_lstm(next_gly, next_bls, next_h_ts, next_c_ts).squeeze(1).detach()
        advantage = np.zeros(batch_size + 1)
        
        td_target = reward + self.gamma * value_new_state * done_mask
        # time step t에 대한 td_target = R_(t+1) + gamma * V(S_(t+1)). 즉 보상과 value  function의 합을 의미함.  

        for t in reversed(range(batch_size)):
            # delta = reward + (self.gamma * value_new_state * done_mask) - value_old_state 후에 delta[t] 로 바꿀 수 있음. --> 최적화?
            delta = reward[t] + (self.gamma * value_new_state[t] * done_mask[t]) - value_old_state[t] # (TD target과 실제 V(S)와의 차이 : TD_error: delta)
            advantage[t] = delta + (self.gamma * self.lmbda * advantage[t + 1] * done_mask[t])

        advantage = torch.tensor(advantage[:batch_size], dtype=torch.float).to(device).detach()
        value_target = advantage[:batch_size] + value_old_state

        return value_target, advantage[:batch_size], td_target


    def update(self):
        clip_param = 0.1 # 0.2 # clip param을 없애도 큰 차이는 없을 수 있다. (안정적, 속도 저하)

        gly, bls, next_gly, next_bls, action, reward, log_prob, done_mask, h_ts, c_ts, next_h_ts, next_c_ts = self.memory.sample()

        gly = gly.view([-1, 21, 79])
        bls = bls.view([-1, 26])
        next_gly = next_gly.view([-1, 21, 79])
        next_bls = next_bls.view([-1, 26])
        reward = reward.reshape(self.flags.num_actors * self.batch_size)
        done_mask = done_mask.reshape(self.flags.num_actors * self.batch_size)
        log_prob = log_prob.reshape(self.flags.num_actors * self.batch_size)
        action = action.reshape(self.flags.num_actors * self.batch_size)
        h_ts = h_ts.view([-1, 128])
        c_ts = c_ts.view([-1, 128])
        next_h_ts = next_h_ts.view([-1, 128])
        next_c_ts = next_c_ts.view([-1, 128])

        returns, advantages, td_target = self.calc_advantage(gly, bls, next_gly, next_bls, reward, done_mask, h_ts, c_ts, next_h_ts, next_c_ts) # batch_size


        PPO_epochs = 7
        actor_losses = []
        critic_losses = []
        ratios = []
        td_targets = []
        returns_ = []

        """  random을 일단 뺌  """
        """
        n = self.batch_size
        arr = np.arange(n)
        np.random.shuffle(arr)
        batch_index = arr[:]  
        batch = torch.LongTensor(batch_index)
         gly_, bls_, action_, old_log_probs, return_, advantage = gly[batch], bls[batch], action[batch], log_prob[batch], returns[batch], advantages[batch]

        """


        for j in range(PPO_epochs):

            returns, advantages, td_target = self.calc_advantage(gly, bls, next_gly, next_bls, reward, done_mask, h_ts, c_ts, next_h_ts, next_c_ts) # batch_size
            # breakpoint()
            gly_, bls_, action_, old_log_probs, return_, advantage = gly, bls, action, log_prob, returns, advantages
            
            dist, _, _ = self.policy.forward_lstm(gly_, bls_, h_ts, c_ts)
            
            new_probs = dist.log_prob(action_)
            
            ratio = (new_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            value = self.value.forward_lstm(gly_, bls_, h_ts, c_ts).float()
            # value_loss = (return_ - value).pow(2).mean() # MSE_loss
            value_loss = F.smooth_l1_loss(value.squeeze() , td_target.detach())
            # value_loss = F.mse_loss(value.squeeze() , td_target.detach())

            # for debugging (tensorboard)
            actor_losses.append(policy_loss.item())
            critic_losses.append(value_loss.item())
            ratios.append(ratio.mean().item())

            td_targets.append(td_target.mean().item())
            returns_.append(returns.mean().item())

            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.policy_optimizer.step()


            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 10.0)
            self.value_optimizer.step()


        
            


        self.memory.clear()

        actor_losses = np.array(actor_losses)
        critic_losses = np.array(critic_losses)
        ratios = np.array(ratios)
        td_targets = np.array(td_targets)
        returns_ = np.array(returns_)
        return [ actor_losses.mean(), critic_losses.mean(), ratios.mean(), td_targets.mean(), returns_.mean()]


    def train(self):
       
        env = self.env 
        e_rewards = [0.0] * self.num_envs
        tot_steps = 0
        steps = [0 for i in range(self.num_envs)]
        step = [1 for i in range(self.num_envs)]

        n_steps = 0
        action_steps = []
        save = 0
        print_freq = 1
        max_step = 150
        state = env.reset() # each reset generates a new environment instance    

        # for tensorboard graph confirm
        loss = []
        done_steps = []
        done_rewards = []
        entropy_loss = []
        actor_loss = []
        critic_loss = []
        ratio = []
        td_target = []
        return_ = []

        # breakpoint()

        while True:
            
            # state = env.reset()

            for mini_step in range(max_step):
                
                tot_steps += 1
                n_steps += 1

                h_t = self.h_t_policy
                c_t = self.c_t_policy
                # copy 확인해보기 
                
                action, dist = self.get_action(state) # action : tensor([2, 2, 0, 0], device='cuda:0') ,  dist : Categorical(probs: torch.Size([4, 4]))

                new_state, reward, done, info =  env.step(action.tolist())
        
                e_rewards = [x + y for x,y in zip(e_rewards, reward)] # num_envs

                log_prob = dist.log_prob(action).tolist()

                """ evaluate """

                
                self.memory.cache(state, new_state, action, reward, log_prob, done, h_t, c_t, self.h_t_policy, self.c_t_policy)
                
                # update
                if mini_step == max_step - 1:
                    losses = self.update()

                    actor_loss.append(losses[0])
                    critic_loss.append(losses[1])
                    ratio.append(losses[2])
                    td_target.append(losses[3])
                    return_.append(losses[4])
                    # breakpoint()

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

            if len(done_rewards) // 100 > print_freq :
                save += 1
                print_freq += 1
                print("************************************************")
                print("mean_steps: {} and tot_steps: {}".format(np.mean(done_steps[-101:-1]), tot_steps))
                print("num_episodes: {} {}".format(len(done_rewards), len(action_steps)))
                print("mean 100 episode reward: {}".format(round(np.mean(done_rewards[-101:-1]), 2)))
                print("************************************************")
                
                if save % 50 == 0:
                    torch.save(self.policy, "PPO_lstm/{}/model".format(self.model_num) + str(save))

                if self.flags.debug == 0:
                    self.writer.add_scalar("mean_reward", round(np.mean(done_rewards[-101:-1]), 2), print_freq)
                    self.writer.add_scalar("mean_steps", np.mean(done_steps[-101:-1]), print_freq)
                    self.writer.add_scalar("Loss/actor_loss", sum(actor_loss)/ len(actor_loss), print_freq)
                    self.writer.add_scalar("Loss/critic_loss", sum(critic_loss)/ len(critic_loss), print_freq)
                    self.writer.add_scalar("Loss/ratio", sum(ratio)/ len(ratio), print_freq)
                    self.writer.add_scalar("Loss/td_target", sum(td_target)/ len(td_target), print_freq)
                    self.writer.add_scalar("Loss/return", sum(return_)/ len(return_), print_freq)

                loss = []
                entropy_loss = []
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

            length = 0
            while not done:
                steps += 1
                tot_steps += 1
                length += 1
                # step
                action,dist = self.get_action(state)


                actions.append(action.item())
                new_state, reward, done, info =  env.step(action.item())
                # env.render("human")

                state = new_state

                e_rewards[-1] += reward
                
                if length > 30:
                    env.render("human")

            # env.render("human")
            
            print(actions, e_rewards[-1], len(actions))
            # env.render("human")

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

        