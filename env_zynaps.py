import numpy as np

from frame import Frame
from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_render
from env_default import default_reset

########################################################################
class ZynapsEnv():
    def __init__(self):
        self.action_space = 18
        self.lives = 3
        self.score = 0
        self.emu = Emulator('./roms/zynaps.z80')
        self.latestFrame = None
        self.viewer = None
        self.viewport = (70,-74,62,-122)
        self.skip_frames = 3 

    def reset(self, skip=0):
        self.lives = 3
        self.score = 0
        default_reset(self.emu, skip)
        next_state = self.latestFrame = default_render(self.emu, self.viewport, self.skip_frames)
        return next_state

    def render(self, viewer):
        if self.latestFrame == None: return
        frame = np.asarray(self.latestFrame)
        arr = np.swapaxes(frame, 0, 2)
        viewer.imshow(arr.astype(np.uint8))

    def step(self, action):
        emu = self.emu
        default_action(emu, action, (Key.W, Key.S, Key.X, Key.C, Key.Q))
        next_state = default_render(emu, self.viewport, self.skip_frames)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        self.latestFrame = next_state
        return (next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        newLives = emu.GetByte(0xE470) - 0x1E
        oldLives = self.lives;
        self.lives = newLives;

        if self.lives == 0: # terminal state
            return True;

        return False;

    def UpdateReward(self):
        newScore = self.ReadScore()
        if newScore == 0: return 0
        reward = newScore - self.score
        self.score = newScore
        return reward

    def ReadScore(self):
        return 10 * (self.emu.GetByte(0xE013) + (self.emu.GetByte(0xE014) * 256))
