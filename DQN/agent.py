import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque
from nle import nethack

from torch.nn import functional as F
from DQN.netDQN import DQN
from DQN.replaybuffer import Replaybuffer

# First, we use state just glyphs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, FLAGS = None):
        
        if FLAGS.mode == "test":
            print("DQN/model/" + FLAGS.model_dir)
            self.policy = torch.load("DQN/model/" + FLAGS.model_dir)
            self.episode = FLAGS.episodes
            self.print_freq = FLAGS.print_freq
        else:   
            # policy network
            self.policy = DQN().to(device)
            # target network
            self.target = DQN().to(device)
        
            # initial optimize
            self.optimizer = torch.optim.Adam(self.policy.parameters())

            self.buffer = Replaybuffer()

            self.gamma = 0.9
            self.batch_size = 32
            self.target_update = 50
            self.episode = FLAGS.episodes
            self.eps_start = FLAGS.eps_start
            self.eps_end = FLAGS.eps_end
            self.print_freq = FLAGS.print_freq
            self.save_freq = FLAGS.save_freq
            eps_fraction = 0.3
            self.eps_timesteps = eps_fraction * float(self.episode)

            # 사용할 함수 
            self.eps_threshold(epi_num = 1)


        
    def eps_threshold(self, epi_num):

        fraction = min(1.0, float(epi_num) / self.eps_timesteps)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def get_action(self, obs):

        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q = self.policy(observed_glyphs,observed_stats)
        
        _, action = q.max(1) # 가장 좋은 action 뽑기 
        return action.item()

    def update(self):
        gly, bls, next_gly, next_bls, action, reward, done = self.buffer.sample(self.batch_size)
        
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

        # env = gym.make(id = "MiniHack-ETest-v0", observation_keys = ("glyphs","chars","colors","specials","blstats","message"))
        env = gym.make(id = "MiniHack-Room-5x5-v0", observation_keys = ("glyphs","chars","colors","specials","blstats","message"))

        e_rewards = [0.0]
        eps_threshold = 0
        tot_steps = 0
        steps = 0

        for epi in range(self.episode):
            done = False
            state = env.reset() # each reset generates a new environment instance    
            
            while not done:
                steps += 1
                tot_steps += 1
                # step
                eps_threshold = self.eps_threshold(epi)
                if random.random() < eps_threshold:
                    action = env.action_space.sample()
                else:
                    action = self.get_action(state)
                
                new_state, reward, done, info =  env.step(action)

                e_rewards[-1] += reward
                # save buffer 
                self.buffer.cache(state, new_state, action, reward, done)
                # update 
                state = new_state

                if self.buffer.len() > self.batch_size:
                    self.update()
                
                # target network update 
                if epi % self.target_update == 0:
                    self.target.load_state_dict(self.policy.state_dict())


            # 한번 episode 시행 후, 
            e_rewards.append(0.0)
            print("Episode: ", epi, "  / step: ", tot_steps )

            #logging
            if len(e_rewards) % self.print_freq == 0 :
                print("************************************************")
                print("mean_steps: {} and tot_steps: {}".format(steps / 25, tot_steps))
                print("num_episodes: {}".format(len(e_rewards)))
                print("mean 100 episode reward: {}".format(round(np.mean(e_rewards[-101:-1]), 2)))
                print("% time spend exploring: {}".format(int(100 * eps_threshold)))
                print("************************************************")

                steps = 0
            #model save 
            if len(e_rewards) % self.save_freq == 0:
                torch.save(self.policy, "DQN/model/model" + str(len(e_rewards)))


    def test(self, random_ex = False):
        pass
        env = gym.make(id = "MiniHack-Room-5x5-v0", observation_keys = ("glyphs","chars","colors","specials","blstats","message"))

        e_rewards = [0.0]
        eps_threshold = 0
        tot_steps = 0

        for epi in range(self.episode):
            done = False
            state = env.reset() # each reset generates a new environment instance
            steps= 0        
            
            while not done:
                steps += 1
                tot_steps += 1

                # step
                if random_ex:
                    eps_threshold = self.eps_threshold(epi)
                    if random.random() < eps_threshold:
                        action = env.action_space.sample()
                    else:
                        action = self.get_action(state)
                else:
                    action = self.get_action(state)
                

                
                new_state, reward, done, info =  env.step(action)
                state = new_state

                e_rewards[-1] += reward
                print("Episode: ", epi, "  / step: ", tot_steps, "\tAction Taken: ", str(action) )
                env.render("human")


            # 한번 episode 시행 후------------------------------------------------------------------------------------
            e_rewards.append(0.0)

            #logging
            if len(e_rewards) % self.print_freq == 0 :
                print("************************************************")
                print("steps: {} and tot_steps: {}".format(steps, tot_steps))
                print("num_episodes: {}".format(len(e_rewards)))
                print("mean 100 episode reward: {}".format(round(np.mean(e_rewards[-101:-1]), 2)))
                # print("% time spend exploring: {}".format(int(100 * eps_threshold)))
                print("************************************************")





        
    


        