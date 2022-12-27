import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque
from nle import nethack

from torch.nn import functional as F
from PPO_discrete.netPPO import Policy, Value
from PPO_discrete.memory import Memory

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


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
                    id = "MiniHack-Room-5x5-v0",

                    observation_keys = ("glyphs","blstats"),
                    actions =  MOVE_ACTIONS,)
        else: 
                        
            self.flags = FLAGS
            embed_dim = 16
            n_layers = 3
            self.num_envs = FLAGS.num_actors
            self.batch_size = 5
            self.gamma = 0.99 # value 값을 줄이면, 
            self.lmbda = 0.95
            self.model_num = FLAGS.model_num
            self.print_freq = self.batch_size
            
            self.env = gym.vector.make(
                id = "MiniHack-Room-5x5-v0",
                # id = "MiniHack-Corridor-R2-v0",
                # id = "MiniHack-MultiRoom-N2-v0",
                observation_keys = ("glyphs","blstats"),
                actions =  MOVE_ACTIONS,
                num_envs =  self.num_envs,
                max_episode_steps= 40,)

            # actor_critic network 
            self.policy = Policy(num_actions= self.env.single_action_space.n, embedding_dim= embed_dim, num_layers= n_layers).to(device)
            self.value = Value(num_actions= self.env.single_action_space.n, embedding_dim= embed_dim, num_layers= n_layers).to(device)
            self.memory = Memory(self.batch_size)

            # initial optimize
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = FLAGS.lr)
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr = FLAGS.lr)

            def lr_lambda_policy(epoch):


                # if epoch < 10000:
                #     lr = FLAGS.lr
                # elif epoch < 20000:
                #     lr = FLAGS.lr * 0.5
                # elif epoch < 60000:
                #     lr = FLAGS.lr * 0.25
                # elif epoch < 100000:
                #     lr = FLAGS.lr * 0.125
                # else:
                #     lr = FLAGS.lr * 0.1

                lr = FLAGS.lr
                if epoch < 10000 * 7 :
                    lr = FLAGS.lr * 0.25
                return lr

            
            def lr_lambda_value(epoch):


                # if epoch < 20000:
                #     lr = FLAGS.lr 
                # elif epoch < 40000:
                #     lr = FLAGS.lr * 0.5 
                # elif epoch < 60000:
                #     lr = FLAGS.lr * 0.25 
                # elif epoch < 100000:
                #     lr = FLAGS.lr * 0.125 
                # else:
                #     lr = FLAGS.lr * 0.1 

                lr = FLAGS.lr
                if epoch < 10000 * 7 :
                    lr = FLAGS.lr * 0.25

                if epoch == 70000:
                    print("lr is change to * 0.25")
                    breakpoint()
             
                return lr

            # self.scheduler_policy = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda_policy)
            # self.scheduler_value = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda_value)

      
            
    def get_action(self, obs):

        if self.flags.mode == 'test':
            observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
            observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)


        else:
            observed_glyphs = torch.from_numpy(obs['glyphs']).float().to(device)
            observed_stats = torch.from_numpy(obs['blstats']).float().to(device)

        with torch.no_grad():
            dist = self.policy.forward(observed_glyphs,observed_stats)  

        if self.flags.mode == 'test':
            return dist 

        return dist.sample(), dist

    
    def get_policy_value(self, obs, new_obs, action_):
        observed_glyphs = torch.from_numpy(obs['glyphs']).float().to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().to(device)
        new_observed_glyphs = torch.from_numpy(new_obs['glyphs']).float().to(device)
        new_observed_stats = torch.from_numpy(new_obs['blstats']).float().to(device)

        with torch.no_grad():
            dist = self.policy.forward(observed_glyphs,observed_stats)  
            new_probs = dist.log_prob(action_)
            value_old_state = self.value.forward(observed_glyphs,observed_stats).squeeze(1).detach() # [600, 1] --> [600]
            value_new_state = self.value.forward(new_observed_glyphs,new_observed_stats).squeeze(1).detach()
        
        return new_probs, value_old_state, value_new_state



    def calc_advantage(self, gly, bls, next_gly, next_bls, reward, done_mask):
        
        batch_size = done_mask.shape[0]
        value_old_state = self.value.forward(gly, bls).squeeze(1).detach() # [600, 1] --> [600]
        value_new_state = self.value.forward(next_gly, next_bls).squeeze(1).detach()
        advantage = np.zeros(batch_size + 1)

        for t in reversed(range(batch_size)):
            # delta = reward + (self.gamma * value_new_state * done_mask) - value_old_state 후에 delta[t] 로 바꿀 수 있음. --> 최적화?
            delta = reward[t] + (self.gamma * value_new_state[t] * done_mask[t]) - value_old_state[t]
            advantage[t] = delta + (self.gamma * self.lmbda * advantage[t + 1] * done_mask[t])

        advantage = torch.tensor(advantage[:batch_size], dtype=torch.float).to(device).detach()
        value_target = advantage[:batch_size] + value_old_state
        
        return value_target, advantage[:batch_size]


    def update(self):
        clip_param = 0.1 # 0.2 # clip param을 없애도 큰 차이는 없을 수 있다. (안정적, 속도 저하)

        gly, bls, next_gly, next_bls, action, reward, log_prob, done_mask = self.memory.sample()
        breakpoint()

        gly = gly.view([-1, 21, 79])
        bls = bls.view([-1, 26])
        next_gly = next_gly.view([-1, 21, 79])
        next_bls = next_bls.view([-1, 26])
        reward = reward.reshape(self.flags.num_actors * self.batch_size)
        done_mask = done_mask.reshape(self.flags.num_actors * self.batch_size)
        log_prob = log_prob.reshape(self.flags.num_actors * self.batch_size)
        action = action.reshape(self.flags.num_actors * self.batch_size)

        returns, advantages = self.calc_advantage(gly, bls, next_gly, next_bls, reward, done_mask) # batch_size


        PPO_epochs = 7
        actor_losses = []
        critic_losses = []
        ratios = []

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
            
            # breakpoint()
            gly_, bls_, action_, old_log_probs, return_, advantage = gly, bls, action, log_prob, returns, advantages
            
            dist = self.policy.forward(gly_, bls_)
            
            new_probs = dist.log_prob(action_)
            
            breakpoint()
            ratio = (new_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            value = self.value.forward(gly_, bls_).float()
            value_loss = (return_ - value).pow(2).mean() # MSE_loss
        
            # for debugging (tensorboard)
            actor_losses.append(policy_loss.item())
            critic_losses.append(value_loss.item())
            ratios.append(ratio.mean().item())


            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.policy_optimizer.step()


            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 10.0)
            self.value_optimizer.step()

            # self.scheduler_policy.step()
            # self.scheduler_value.step()
        
            


        self.memory.clear()

        # losses = np.array(losses)
        actor_losses = np.array(actor_losses)
        critic_losses = np.array(critic_losses)
        ratios = np.array(ratios)

        return [ actor_losses.mean(), critic_losses.mean(), ratios.mean()]


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
        max_step = 40
        state = env.reset() # each reset generates a new environment instance    

        # for tensorboard graph confirm
        loss = []
        done_steps = []
        done_rewards = []
        entropy_loss = []
        actor_loss = []
        critic_loss = []
        ratio = []


        while True:

            for mini_step in range(max_step):
                
                tot_steps += 1
                n_steps += 1

                action, dist = self.get_action(state) # action : tensor([2, 2, 0, 0], device='cuda:0') ,  dist : Categorical(probs: torch.Size([4, 4]))

                new_state, reward, done, info =  env.step(action.tolist())
        
                e_rewards = [x + y for x,y in zip(e_rewards, reward)] # num_envs

                log_prob = dist.log_prob(action).tolist()

                """ evaluate """

                
                self.memory.cache(state, new_state, action, reward, log_prob, done)
                
                # update
                if mini_step == max_step - 1:
                    losses = self.update()

                    actor_loss.append(losses[0])
                    critic_loss.append(losses[1])
                    ratio.append(losses[2])


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
                    torch.save(self.policy, "PPO_discrete/{}/model".format(self.model_num) + str(save))

                if self.flags.debug == 0:
                    self.writer.add_scalar("mean_reward", round(np.mean(done_rewards[-101:-1]), 2), print_freq)
                    self.writer.add_scalar("mean_steps", np.mean(done_steps[-101:-1]), print_freq)
                    self.writer.add_scalar("Loss/actor_loss", sum(actor_loss)/ len(actor_loss), print_freq)
                    self.writer.add_scalar("Loss/critic_loss", sum(critic_loss)/ len(critic_loss), print_freq)
                    self.writer.add_scalar("Loss/ratio", sum(ratio)/ len(ratio), print_freq)

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


            while not done:
                steps += 1
                tot_steps += 1
                # step
                action = self.get_action(state)


                actions.append(action)
                new_state, reward, done, info =  env.step(action)
                # env.render("human")

                state = new_state

                e_rewards[-1] += reward

            
            print(actions, e_rewards[-1], len(actions))
            env.render("human")

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

        