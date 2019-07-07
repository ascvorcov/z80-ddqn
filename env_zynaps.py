import numpy as np

from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_render

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

    def reset(self):
        self.lives = 3
        self.score = 0
        emu = self.emu
        emu.Reset()
        next_state = self.latestFrame = default_render(emu, self.viewport)
        return next_state

    def render(self, viewer):
        if self.latestFrame == None: return
        frame = np.asarray(self.latestFrame)
        arr = np.swapaxes(frame, 0, 2)
        viewer.imshow(arr.astype(np.uint8))

    def step(self, action):
        emu = self.emu
        default_action(emu, action, (Key.W, Key.S, Key.X, Key.C, Key.Q))

        next_state = default_render(emu, self.viewport)
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
