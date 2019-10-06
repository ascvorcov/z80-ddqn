import os.path
import numpy as np

from env_riverraid import RiverRaidEnv
from env_zynaps import ZynapsEnv
from env_renegade import RenegadeEnv
from env_krakout import KrakoutEnv
from env_barbarian import BarbarianEnv

########################################################################
class MainGymWrapper():
    def __init__(self, name, skip):
        self.name = name
        self.skip = skip
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
        return self.env.reset(np.random.randint(self.skip) if self.skip > 0 else 0)

    def render(self, renderer):
        return self.env.render(renderer)

    def step(self, action):
        return self.env.step(action)
