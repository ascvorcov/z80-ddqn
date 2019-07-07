import os.path
import numpy as np

from env_riverraid import RiverRaidEnv
from env_zynaps import ZynapsEnv
from env_renegade import RenegadeEnv

from image_viewer import SimpleImageViewer

########################################################################
class MainGymWrapper():
    def __init__(self, name):
        self.name = name
        self.viewer = None
        if name == 'Riverraid':
            self.env = RiverRaidEnv()
        elif name == 'Zynaps':
            self.env = ZynapsEnv()
        elif name == 'Renegade':
            self.env = RenegadeEnv()

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        return self.env.reset()

    def render(self):
        if self.viewer == None:
            self.viewer = SimpleImageViewer()
        return self.env.render(self.viewer)

    def step(self, action):
        return self.env.step(action)

########################################################################
class ExolonEnv():
    def __init__(self):
        pass

########################################################################
class RTypeEnv():
    def __init__(self):
        pass

