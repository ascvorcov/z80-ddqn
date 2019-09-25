import numpy as np

from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_next_frame
from env_default import default_render
from env_default import default_reset

########################################################################
class ZynapsEnv():
    def __init__(self):
        self.action_space = 18
        self.lives = 3
        self.score = 0
        self.emu = Emulator("./roms/zynaps.z80")
        self.latest_frame = None
        self.next_state = None
        self.viewport = (70,-74,62,-122)
        self.skip_frames = 3 

    def reset(self, skip=0):
        self.lives = 3
        self.score = 0
        default_reset(self.emu, skip)
        self.latest_frame, self.next_state = default_next_frame(self.emu, self.viewport, self.skip_frames)
        return self.next_state

    def render(self, viewer, render_what="state"):
        default_render(viewer, self.next_state if render_what == "state" else self.latest_frame)

    def step(self, action):
        emu = self.emu
        default_action(emu, action, (Key.W, Key.S, Key.X, Key.C, Key.Q))
        self.latest_frame, self.next_state = default_next_frame(emu, self.viewport, self.skip_frames)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        return (self.next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        new_lives = emu.GetByte(0xE470) - 0x1E
        old_lives = self.lives;
        self.lives = new_lives;

        if self.lives == 0: # terminal state
            return True;

        return False;

    def UpdateReward(self):
        new_score = self.ReadScore()
        if new_score == 0: return 0
        reward = new_score - self.score
        self.score = new_score
        return reward

    def ReadScore(self):
        return 10 * (self.emu.GetByte(0xE013) + (self.emu.GetByte(0xE014) * 256))
