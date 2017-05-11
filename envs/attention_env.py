from . attention_board import AttentionBoard
# from . board_display import *

from PIL import Image

from collections import deque

import numpy as np
import sys
import math

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

SIZE=48
COMPLEX=False
SUM_REWARD=False
STATIC=True

class AttentionEnv:
    metadata = {'render.modes': ['human', 'array']}

    def __init__(self, complex=False, sum_reward=False, static=False):
        if complex:
            self.board = AttentionBoard2(SIZE)
        else:
            self.board = AttentionBoard(SIZE)
        self.display = None
        self.reward = 0.
        self._sum_reward = sum_reward
        
        self.action_space = 9
        self.observation_space = np.ndarray((SIZE,SIZE))

        self.frames = deque([])

        if static: self.mode = 'static'
        else: self.mode=None
    
    def preprocess(self, frame):
        frame = frame / 255
        return torch.from_numpy(frame).float()

    def step(self, action):
        step_reward = self.board.step(action)
        if self._sum_reward:
            # if step_reward > 0 and self.reward < 0:
                # self.reward = self.reward / 2
            if step_reward < 0: step_reward = -1
            self.reward += step_reward
        else:
            self.reward = step_reward
        obs, done = self.board.next(mode=self.mode)
        self.frames.popleft()
        self.frames.append(self.preprocess(obs))
        state = torch.stack(self.frames, dim=2).numpy()
        return (state, self.reward, done, None)

    def reset(self):
        self.board = AttentionBoard(SIZE)
        self.reward = 0.
        frame = self.preprocess(self.board.image())
        self.frames = deque([frame] * 4)
        return torch.stack(self.frames, dim=2).numpy()

    def render(self, mode='human', close=False):
        return torch.stack(self.frames, dim=2).numpy()
