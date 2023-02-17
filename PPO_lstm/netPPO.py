import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque
from nle import nethack

from torch.nn import functional as F
from torch.distributions import Categorical

np.set_printoptions(threshold=np.inf, linewidth=np.inf) #for debuger


# hidden size = 64 



class Crop(nn.Module):
    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)[
                     None, :
                     ].expand(self.height_target, -1)
        height_grid = self._step_to_range(2 / (self.height - 1), height_target)[
                      :, None
                      ].expand(-1, self.width_target)

        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def _step_to_range(self, delta, num_steps):
        return delta * torch.arange(-num_steps // 2, num_steps // 2)

    def forward(self, inputs, coordinates):
        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
                .squeeze(1)
                .long()
        )



class Policy(nn.Module):
    def __init__(self, embedding_dim=32, crop_dim=9, num_layers=5, num_actions = 8):
        super(Policy, self).__init__()

        self.glyph_shape = (21, 79)
        """JHJ"""
        # self.blstats_shape = 26
        self.blstats_shape = 26

        self.num_actions = num_actions
        self.h = self.glyph_shape[0]
        self.w = self.glyph_shape[1]
        self.k_dim = embedding_dim
        self.h_dim = 128

        self.glyph_crop = Crop(self.h, self.w, crop_dim, crop_dim)
        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim
        F = 3
        S = 1
        P = 1
        M = 16
        Y = 8
        L = num_layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )
        
        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim

        # CNN crop model.
        """JHJ"""
        out_dim = 0
        out_dim += crop_dim**2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_shape, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
                nn.Linear(self.h_dim, self.num_actions),
                nn.Softmax(dim = 1) # agent가 각 행동을 취할 probability가 나오게 됨. 
        )

        self.lstm = nn.LSTMCell(self.h_dim, self.h_dim)
    
    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))

        return out.reshape(x.shape + (-1,))
    
    def forward_lstm(self, observed_glyphs, observed_stats, h_t, c_t):
        # assert 1 == 3
        B = observed_glyphs.shape[0]
        blstats_emb = self.embed_blstats(observed_stats)
        reps = [blstats_emb]

        """JHJ"""
        reps = []

        coordinates = observed_stats[:, :2]
        observed_glyphs = observed_glyphs.long()
        crop = self.glyph_crop(observed_glyphs, coordinates)

        crop_emb = self._select(self.embed, crop)
        crop_emb = crop_emb.transpose(1, 3)
        crop_rep = self.extract_crop_representation(crop_emb)
        crop_rep = crop_rep.view(B, -1)
        reps.append(crop_rep)
        
      
        st = torch.cat(reps, dim=1)

        st = self.fc(st)

        h_t, c_t = self.lstm(st, (h_t, c_t))
        st = self.actor(h_t)
        actor = st 
        dist = Categorical(actor)

        mode = "train" # PPO test 에서는 return 값이 가장 큰 것을 고르도록(항상 max 선택)
        if mode == "test":
            return self.select_action(actor)
        
        return dist, h_t, c_t
    
    def select_action(self, actor):
        actor = actor.tolist()[0]
        return actor.index(max(actor))
        
 


""" ***********    value network    ********************************************************************** """

class Value(nn.Module):
    def __init__(self, embedding_dim=32, crop_dim=9, num_layers=5, num_actions = 8):
        super(Value, self).__init__()

        self.glyph_shape = (21, 79)

        self.blstats_shape = 26

        self.num_actions = num_actions
        self.h = self.glyph_shape[0]
        self.w = self.glyph_shape[1]
        self.k_dim = embedding_dim
        self.h_dim = 128

        self.glyph_crop = Crop(self.h, self.w, crop_dim, crop_dim)
        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim
        F = 3
        S = 1
        P = 1
        M = 16
        Y = 8
        L = num_layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )
        
        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim


        # CNN crop model.
        """JHJ"""
        out_dim = 0 
        out_dim += crop_dim**2 * Y

        # print(out_dim)

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_shape, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )


        self.fc_critic = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )
    
        self.critic = nn.Linear(self.h_dim, 1)

        self.lstm = nn.LSTMCell(self.h_dim, self.h_dim)

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))

        return out.reshape(x.shape + (-1,))
    
    def forward(self, observed_glyphs, observed_stats):

        B = observed_glyphs.shape[0]

        blstats_emb = self.embed_blstats(observed_stats)
        reps = [blstats_emb]

        """JHJ"""
        reps = []

        coordinates = observed_stats[:, :2]
        observed_glyphs = observed_glyphs.long()
        crop = self.glyph_crop(observed_glyphs, coordinates)

        crop_emb = self._select(self.embed, crop)
        crop_emb = crop_emb.transpose(1, 3)
        crop_rep = self.extract_crop_representation(crop_emb)
        crop_rep = crop_rep.view(B, -1)
        reps.append(crop_rep)
        
      
        st = torch.cat(reps, dim=1)

        st = self.fc_critic(st)
        critic = self.critic(st)

        return critic
    

    def forward_lstm(self, observed_glyphs, observed_stats, h_t, c_t):

        B = observed_glyphs.shape[0]

        blstats_emb = self.embed_blstats(observed_stats)
        reps = [blstats_emb]

        """JHJ"""
        reps = []

        coordinates = observed_stats[:, :2]
        observed_glyphs = observed_glyphs.long()
        crop = self.glyph_crop(observed_glyphs, coordinates)

        crop_emb = self._select(self.embed, crop)
        crop_emb = crop_emb.transpose(1, 3)
        crop_rep = self.extract_crop_representation(crop_emb)
        crop_rep = crop_rep.view(B, -1)
        reps.append(crop_rep)
        
      
        st = torch.cat(reps, dim=1)

        st = self.fc_critic(st)
        h_t, c_t = self.lstm(st, (h_t, c_t))
        critic = self.critic(h_t)

        return critic