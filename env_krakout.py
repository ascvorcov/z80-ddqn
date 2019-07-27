import numpy as np

from emulator import Key
from emulator import Emulator
from env_default import default_action 
from env_default import default_render
from env_default import default_reset

########################################################################
class KrakoutEnv():
    def __init__(self):
        self.action_space = 5
        self.lives = 3
        self.score = 0
        self.emu = Emulator('./roms/krakout.z80')
        self.latestFrame = None
        self.viewer = None
        self.viewport = (84,-60,104,-80)
        self.skip_frames = 0 

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

        next_state = default_render(emu, self.viewport, self.skip_frames)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        self.latestFrame = next_state
        return (next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        newLives = emu.GetByte(0x8E9D)
        oldLives = self.lives;
        self.lives = newLives;

        if self.lives == 0: # terminal state
            return True;

        return False;

    def UpdateReward(self):
        newScore = self.ReadScore()
        if newScore <= 0: return 0
        reward = newScore - self.score
        if reward < 0: return 0
        self.score = newScore
        return reward

    def ReadScore(self):
        return 10 * (self.emu.GetByte(0x5B71) + (self.emu.GetByte(0x5B72) * 256))
