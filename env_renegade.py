import numpy as np

from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_next_frame
from env_default import default_render
from env_default import default_reset

########################################################################
class RenegadeEnv():
    def __init__(self):
        self.action_space = 18
        self.lives = 3
        self.score = 0
        self.emu = Emulator("./roms/renegade.z80")
        self.latest_frame = None
        self.next_state = None
        self.viewport = (50,-94,92,-92)
        self.skip_frames = 2

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
        default_action(emu, action, (Key.I, Key.K, Key.J, Key.L, Key.Q))

        self.latest_frame, self.next_state = default_next_frame(emu, self.viewport, self.skip_frames)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        return (self.next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        new_lives = emu.GetByte(0x5B2F) - 1
        old_lives = self.lives;
        self.lives = new_lives;

        if emu.GetByte(0xBF13) == 0x0F: # even more strict condition - consider knocked down as terminal
            return True
        if self.lives == 1: # consider loss of 1 life a terminal state
            return True

        return False

    def UpdateReward(self):
        new_score = self.ReadScore()
        if new_score == 0: return 0
        reward = new_score - self.score
        self.score = new_score
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
