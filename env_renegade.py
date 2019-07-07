import numpy as np

from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_render

########################################################################
class RenegadeEnv():
    def __init__(self):
        self.action_space = 18
        self.lives = 3
        self.score = 0
        self.emu = Emulator('./roms/renegade.z80')
        self.latestFrame = None
        self.viewer = None
        self.viewport = (50,-94,92,-92)

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
        default_action(emu, action, (Key.I, Key.K, Key.J, Key.L, Key.Q))

        next_state = default_render(emu, self.viewport)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        self.latestFrame = next_state
        return (next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        newLives = emu.GetByte(0x5B2F) - 1
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
        emu = self.emu
        d1 = emu.GetByte(0x9E3D) - 0x30
        d2 = emu.GetByte(0x9E3E) - 0x30
        d3 = emu.GetByte(0x9E3F) - 0x30
        d4 = emu.GetByte(0x9E40) - 0x30
        d5 = emu.GetByte(0x9E41) - 0x30
        d6 = emu.GetByte(0x9E42) - 0x30
        return d1 * 100000 + d2 * 10000 + d3 * 1000 + d4 * 100 + d5 * 10 + d6
