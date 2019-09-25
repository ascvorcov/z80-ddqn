import datetime
from logger import Logger

class BaseGameModel:

    def __init__(self, game_name, mode_name, logger_path, input_shape, action_space):
        self.initial_run = 0
        self.initial_total_step = 0
        self.action_space = action_space
        self.input_shape = input_shape
        self.logger = Logger(game_name + " " + mode_name, logger_path)

    def save_run(self, score, step, run):
        self.logger.add_score(score)
        self.logger.add_step(step)
        self.logger.add_run(run)

    def save(self, run, total_step):
        pass

    def get_move(self, state):
        pass

    def move(self, state):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def step_update(self, total_step):
        pass

    def _get_date(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

