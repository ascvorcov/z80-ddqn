import time
import argparse
import numpy as np
from ddqn_game_model import DDQNTrainer, DDQNSolver
from gym_wrappers import MainGymWrapper
from frame import FRAME_SIZE

FRAMES_IN_OBSERVATION = 4
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


class Main:

    def __init__(self):
        game_name, game_mode, render, total_step_limit, total_run_limit, clip, skip = self._args()
        env = MainGymWrapper(game_name, skip)
        self._main_loop(self._game_model(game_mode, game_name, env.action_space), env, render, total_step_limit, total_run_limit, clip)

    def _main_loop(self, game_model, env, render, total_step_limit, total_run_limit, clip):
        run = 0
        total_step = 0
        viewer = None

        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print ("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1
            current_state = env.reset()
            step = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print ("Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1
                step += 1

                if render:
                    env.render()

                action = game_model.move(current_state)
                next_state, reward, terminal = env.step(action)
                if clip:
                    reward = np.sign(reward)
                score += reward
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state
                game_model.step_update(total_step)
                time.sleep(1)
                if terminal:
                    print('score:%d\tsteps:%d\ttotal:%d\trun:%d' % (score, step, total_step, run))
                    game_model.save_run(score, step, run)
                    break

    def _args(self):
        parser = argparse.ArgumentParser()
        available_games = ['Krakout','Riverraid','Renegade','Zynaps']
        parser.add_argument("-g", "--game", help="Choose from available games: " + str(available_games) + ". Default is 'Riverraid'.", default="Riverraid")
        parser.add_argument("-m", "--mode", help="Choose from available modes: ddqn_train, ddqn_test. Default is 'ddqn_train'.", default="ddqn_train")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.", default=False, type=bool)
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit", help="Choose after how many runs we should stop. Default is None (no limit).", default=None, type=int)
        parser.add_argument("-c", "--clip", help="Choose whether we should clip rewards to (0, 1) range. Default is 'False'", default=False, type=bool)
        parser.add_argument("-s", "--skip", help="Max random number of frames to skip on reset. Default is 0", default=0, type=int)
        args = parser.parse_args()
        game_mode = args.mode
        game_name = args.game
        render = args.render
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        clip = args.clip
        skip = args.skip
        print ("Selected game: " + str(game_name))
        print ("Selected mode: " + str(game_mode))
        print ("Should render: " + str(render))
        print ("Should clip: " + str(clip))
        print ("Should skip frames: " + str(skip))
        print ("Total step limit: " + str(total_step_limit))
        print ("Total run limit: " + str(total_run_limit))
        return game_name, game_mode, render, total_step_limit, total_run_limit, clip, skip

    def _game_model(self, game_mode,game_name, action_space):
        if game_mode == "ddqn_train":
            return DDQNTrainer(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ddqn_test":
            return DDQNSolver(game_name, INPUT_SHAPE, action_space)
        else:
            print ("Unrecognized mode. Use --help")
            exit(1)


if __name__ == "__main__":
    Main()
