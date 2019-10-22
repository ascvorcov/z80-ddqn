import numpy as np
import shutil
import os
from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_next_frame
from env_default import default_reset

########################################################################
class RaidersEnv():
    def __init__(self):
        self.action_space = 6
        self.lives = 3
        self.score = 0
        self.emu = Emulator("./roms/raiders.z80")
        self.latest_frame = None
        self.next_state = None
        self.viewport = (85,90)
        self.skip_frames = 1

    def reset(self, skip=0):
        self.lives = 3
        self.score = 0
        default_reset(self.emu, skip)
        self.latest_frame, self.next_state = default_next_frame(self.emu, self.viewport, self.skip_frames, filter_image=False)
        return self.next_state

    def render(self, renderer):
        renderer.render(self.next_state, self.latest_frame)

    def step(self, action):
        emu = self.emu
        l,r,f = (Key.D1, Key.D2, Key.Space)

        emu.KeyUp(l)
        emu.KeyUp(r)
        emu.KeyUp(f)
        if   action == 0: 
            pass
        elif action == 1: #left
            emu.KeyDown(l)
        elif action == 2: #right
            emu.KeyDown(r)
        elif action == 3: #fire
            emu.KeyDown(f)
        elif action == 4: #leftfire
            emu.KeyDown(l)
            emu.KeyDown(f)
        elif action == 5: #rightfire
            emu.KeyDown(r)
            emu.KeyDown(f)
        else: raise Exception("Error")

        self.latest_frame, self.next_state = default_next_frame(emu, self.viewport, self.skip_frames)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        return (self.next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        new_lives = emu.GetByte(0x6023)
        old_lives = self.lives;
        self.lives = new_lives;

        if new_lives < old_lives:
            return True;

        return False;

    def UpdateReward(self):
        new_score = self.ReadScore()
        if new_score == 0: return 0
        reward = new_score - self.score
        self.score = new_score
        return reward

    def ReadScore(self):
        return 10 * (self.emu.GetByte(0x601B) + (self.emu.GetByte(0x601C) * 256))
