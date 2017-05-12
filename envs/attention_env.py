from . attention_board import AttentionBoard
from . board_display import *

from collections import deque

from PIL import Image

import numpy as np
import sys
import math

SIZE=24

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
        self.frames = None

        self.observation_space = np.ndarray((SIZE,SIZE))
        self.action_space = 5

        if static: self.mode = 'static'
        else: self.mode=None
    
    def stack_frames(self):
        return np.stack(list(self.frames), 0).astype('float32')

    def step(self, action):
        step_reward = self.board.step(action)
        if self._sum_reward:
            # if step_reward > 0 and self.reward < 0:
                # self.reward = self.reward /
            self.reward += step_reward
        else:
            self.reward = step_reward
        obs, done = self.board.next(mode=self.mode)
        self.frames.popleft()
        self.frames.append(obs)
        state = self.stack_frames()
        return (obs, self.reward, done, None)

    def reset(self):
        self.board = AttentionBoard(SIZE)
        self.reward = 0.
        frame = self.board.image()
        self.frames = deque([frame]*3)
        return self.stack_frames()

    def render(self, mode='human', close=False):
        if mode == 'human':
            if self.display == None:
                self.display = BoardDisplay(SIZE, SIZE)
            self.display.render_update(self.board)
        return self.stack_frames()
