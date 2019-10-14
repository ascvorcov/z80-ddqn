import numpy as np

from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_next_frame
from env_default import default_reset

########################################################################
class BarbarianEnv():
    def __init__(self):
        self.action_space = 18
        self.score = 0
        self.emu = Emulator("./roms/barbarian.z80")
        self.latest_frame = None
        self.next_state = None
        self.viewport = (80,-64,64,-120) #u,d,l,r
        self.skip_frames = 2

    def reset(self, skip=0):
        self.score = 0
        default_reset(self.emu, skip)
        self.latest_frame, self.next_state = default_next_frame(self.emu, self.viewport, self.skip_frames)
        return self.next_state

    def render(self, renderer):
        renderer.render(self.next_state, self.latest_frame)

    def step(self, action):
        emu = self.emu
        default_action(emu, action, (Key.I, Key.K, Key.J, Key.L, Key.Q))
        self.latest_frame, self.next_state = default_next_frame(emu, self.viewport, self.skip_frames)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        return (self.next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        enemy_score = emu.GetByte(0xB97E)

        if enemy_score > 0: # enemy scored one hit - terminal state
            if self.score > 2000:
                print('nice score!')
                self.reset()
                exit(0)
            return True;

        return False;

    def UpdateReward(self):
        new_score = self.ReadScore()
        if new_score == 0: return 0
        reward = new_score - self.score
        self.score = new_score
        return reward

    def ReadScore(self):
        return self.emu.GetByte(0xB97B) + (self.emu.GetByte(0xB97C) * 256)
