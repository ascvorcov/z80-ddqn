import os.path
import numpy as np

from env_riverraid import RiverRaidEnv
from env_zynaps import ZynapsEnv
from env_renegade import RenegadeEnv
from env_krakout import KrakoutEnv

from image_viewer import SimpleImageViewer

########################################################################
class MainGymWrapper():
    def __init__(self, name, skip):
        self.name = name
        self.skip = skip
        self.viewer = None
        if name == 'Riverraid':
            self.env = RiverRaidEnv()
        elif name == 'Zynaps':
            self.env = ZynapsEnv()
        elif name == 'Renegade':
            self.env = RenegadeEnv()
        elif name == 'Krakout':
            self.env = KrakoutEnv()

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        return self.env.reset(np.random.randint(self.skip) if self.skip > 0 else 0)

    def render(self):
        if self.viewer == None:
            self.viewer = SimpleImageViewer()
        return self.env.render(self.viewer)

    def step(self, action):
        return self.env.step(action)
