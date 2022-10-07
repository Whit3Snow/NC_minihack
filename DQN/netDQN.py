import gym
import torch
import minihack 
from torch import nn
import random, numpy as np
from collections import deque
from nle import nethack

from torch.nn import functional as F

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

class DQN(nn.Module):
    def __init__(self, embedding_dim=32, crop_dim=9, num_layers=5, num_actions = 8):
        super(DQN, self).__init__()

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
        # CNN over full glyph map
        out_dim += self.h * self.w * Y

        # CNN crop model.
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
            nn.ReLU(),
            nn.Linear(self.h_dim, self.num_actions)
        )
    
    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))

        return out.reshape(x.shape + (-1,))
    
    def forward(self, observed_glyphs, observed_stats):
        # assert 1 == 3
        B = observed_glyphs.shape[0]
        blstats_emb = self.embed_blstats(observed_stats)
        reps = [blstats_emb]

        coordinates = observed_stats[:, :2]
        observed_glyphs = observed_glyphs.long()
        crop = self.glyph_crop(observed_glyphs, coordinates)

        crop_emb = self._select(self.embed, crop)
        crop_emb = crop_emb.transpose(1, 3)
        crop_rep = self.extract_crop_representation(crop_emb)
        crop_rep = crop_rep.view(B, -1)
        reps.append(crop_rep)
        
        glyphs_emb = self._select(self.embed, observed_glyphs)
        glyphs_emb = glyphs_emb.transpose(1, 3)
        glyphs_rep = self.extract_representation(glyphs_emb)
        glyphs_rep = glyphs_rep.view(B, -1)
        reps.append(glyphs_rep)
        
        st = torch.cat(reps, dim=1)
        st = self.fc(st)
        # print("in forward")
        return st
