import numpy as np

from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_next_frame
from env_default import default_reset

########################################################################
class KrakoutEnv():
    def __init__(self):
        self.action_space = 5
        self.lives = 3
        self.score = 0
        self.emu = Emulator("./roms/krakout.z80")
        self.latest_frame = None
        self.next_state = None
        self.viewport = (84,-60,104,-80)
        self.skip_frames = 0 

    def reset(self, skip=0):
        self.lives = 3
        self.score = 0
        default_reset(self.emu, skip)
        self.latest_frame, self.next_state = default_next_frame(self.emu, self.viewport, self.skip_frames)
        return self.next_state

    def render(self, renderer):
        renderer.render(self.next_state, self.latest_frame)

    def step(self, action):
        emu = self.emu
        u,d,f = (Key.P, Key.L, Key.Space)

        emu.KeyUp(u)
        emu.KeyUp(d)
        emu.KeyUp(f)
        if   action == 0: 
            pass
        elif action == 1: #up
            emu.KeyDown(u)
        elif action == 2: #down
            emu.KeyDown(d)
        elif action == 3: #fire
            emu.KeyDown(f)
        elif action == 4: #upfire
            emu.KeyDown(u)
            emu.KeyDown(f)
        elif action == 5: #downfire
            emu.KeyDown(d)
            emu.KeyDown(f)
        else: raise Exception("Error")

        self.latest_frame, self.next_state = default_next_frame(emu, self.viewport, self.skip_frames)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        return (self.next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        new_lives = emu.GetByte(0x8E9D)
        old_lives = self.lives;
        self.lives = new_lives;

        if new_lives < old_lives: # terminal state
            return True;

        return False;

    def UpdateReward(self):
        new_score = self.ReadScore()
        if new_score <= 0: return 0
        reward = new_score - self.score
        if reward < 0: return 0
        self.score = new_score
        return reward

    def ReadScore(self):
        emu = self.emu
        d1 = emu.GetByte(0xB676) - 0x30
        d2 = emu.GetByte(0xB677) - 0x30
        d3 = emu.GetByte(0xB678) - 0x30
        d4 = emu.GetByte(0xB679) - 0x30
        d5 = emu.GetByte(0xB67A) - 0x30
        d6 = emu.GetByte(0xB67B) - 0x30
        d7 = emu.GetByte(0xB67C) - 0x30
        return 10 * (d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7)
