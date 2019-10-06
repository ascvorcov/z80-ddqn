import os.path
import numpy as np

from env_riverraid import RiverRaidEnv
from env_zynaps import ZynapsEnv
from env_renegade import RenegadeEnv
from env_krakout import KrakoutEnv
from env_barbarian import BarbarianEnv

from image_viewer import SimpleImageViewer

########################################################################
class MainGymWrapper():
    def __init__(self, name, render_mode, skip):
        self.name = name
        self.skip = skip
        self.viewer = None
        self.render_mode = render_mode
        if name == "Riverraid":
            self.env = RiverRaidEnv()
        elif name == "Zynaps":
            self.env = ZynapsEnv()
        elif name == "Renegade":
            self.env = RenegadeEnv()
        elif name == "Krakout":
            self.env = KrakoutEnv()
        elif name == "Barbarian":
            self.env = BarbarianEnv()

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        return self.env.reset(self.render_mode, np.random.randint(self.skip) if self.skip > 0 else 0)

    def render(self):
        if self.render_mode == 0: return
        if self.viewer == None:
            self.viewer = SimpleImageViewer()
        return self.env.render(self.viewer, self.render_mode)

    def step(self, action):
        return self.env.step(action)
