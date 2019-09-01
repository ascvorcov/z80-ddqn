import numpy as np

from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_render
from env_default import default_reset

########################################################################
class RiverRaidEnv():
    def __init__(self):
        self.action_space = 18
        self.lives = 3
        self.score = 0
        self.lookup = {0x4A: 0,0x08: 1,0x02: 2,0x0C: 3,0x28: 4,0x04: 7,0x3C: 8,0x42: 9}
        self.emu = Emulator('./roms/riverraid.z80')
        self.latestFrame = None
        self.viewer = None
        self.viewport = (70,-74,92,-92)
        self.skip_frames = 2 

    def reset(self, skip=0):
        self.lives = 3
        self.score = 0
        default_reset(self.emu, skip)
        self.emu.KeyDown(Key.Space) # need to press fire to start
        next_state = self.latestFrame = default_render(self.emu, self.viewport, self.skip_frames)
        return next_state

    def render(self, viewer):
        if self.latestFrame == None: return
        frame = np.asarray(self.latestFrame)
        arr = np.swapaxes(frame, 0, 2)
        viewer.imshow(arr.astype(np.uint8))

    def step(self, action):
        emu = self.emu
        default_action(emu, action, (Key.D2, Key.W, Key.O, Key.P, Key.Space))
        next_state = default_render(emu, self.viewport, self.skip_frames)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        self.latestFrame = next_state
        return (next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        newLives = emu.GetByte(0x923B)
        oldLives = self.lives;
        self.lives = newLives;

        speed = emu.GetByte(0x5F64) # 0 - stopped/crashed, 1,2,4 - slow/normal/fast
        if speed == 0: # player died
            return True; # do not count lives, consider death a terminal state

        return False;

    def UpdateReward(self):
        newScore = self.ReadScore()
        if newScore == 0: return 0
        reward = newScore - self.score
        self.score = newScore
        return reward

    def ReadScore(self):
        # read score directly from screen
        emu = self.emu
        b1 = emu.GetByte(0x53E5)
        b2 = emu.GetByte(0x53E6)
        b3 = emu.GetByte(0x53E7)
        b4 = emu.GetByte(0x53E8)
        b5 = emu.GetByte(0x53E9)
        b6 = emu.GetByte(0x53EA)
        b7 = emu.GetByte(0x53EB)
        
        h1 = emu.GetByte(0x54E5)
        h2 = emu.GetByte(0x54E6)
        h3 = emu.GetByte(0x54E7)
        h4 = emu.GetByte(0x54E8)
        h5 = emu.GetByte(0x54E9)
        h6 = emu.GetByte(0x54EA)
        h7 = emu.GetByte(0x54EB)
        
        d1 = self.GetDigit(b1,h1);
        d2 = self.GetDigit(b2,h2);
        d3 = self.GetDigit(b3,h3);
        d4 = self.GetDigit(b4,h4);
        d5 = self.GetDigit(b5,h5);
        d6 = self.GetDigit(b6,h6);
        d7 = self.GetDigit(b7,h7);
        
        return d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7;

    def GetDigit(self, mem1, mem2):
        if mem1 in self.lookup:
            return self.lookup[mem1]
        elif mem2 == 2 and mem1 == 0x7C:
            return 5
        elif mem2 == 0x42 and mem1 == 0x7C:
            return 6
        return 0