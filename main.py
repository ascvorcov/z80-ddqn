import keyboard
import time
import argparse
import numpy as np
from ddqn_game_model import DDQNTrainer, DDQNSolver
from gym_wrappers import MainGymWrapper
from frame import FRAME_SIZE
from renderer import get_renderer

FRAMES_IN_OBSERVATION = 4
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)

class Main:
    def __init__(self):
        game_name, game_mode, model_name, render_mode, total_step_limit, total_run_limit, clip, skip = self._args()
        env = MainGymWrapper(game_name, skip)
        game_model = self._game_model(game_mode, model_name, game_name, env.action_space)
        renderer = get_renderer(render_mode, game_model.ddqn)

        self._main_loop(game_model, env, renderer, total_step_limit, total_run_limit, clip)

    def _main_loop(self, game_model, env, renderer, total_step_limit, total_run_limit, clip):
        run = game_model.initial_run
        total_step = game_model.initial_total_step
        viewer = None

        if run != 0:
            print ("Continue from run " + str(run))

        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print ("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1
            renderer.reset()
            current_state = env.reset()
            step = 0
            score = 0
            while True:
                if keyboard.is_pressed("q"):
                    print ("Quit button pressed, saving state")
                    game_model.save(run, total_step)
                    exit(0)

                if total_step >= total_step_limit:
                    print ("Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1
                step += 1

                env.render(renderer)

                action = game_model.move(current_state)
                next_state, reward, terminal = env.step(action)
                if clip:
                    reward = np.sign(reward)
                score += reward
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state
                game_model.step_update(total_step)
                if terminal:
                    print("score:%d\tsteps:%d\ttotal:%d\trun:%d" % (score, step, total_step, run))
                    game_model.save_run(score, step, run)
                    break

    def _args(self):
        parser = argparse.ArgumentParser()
        available_games = ["Krakout","Riverraid","Raiders","Renegade","Xecutor","Barbarian"]
        parser.add_argument("-g", "--game", help="Choose from available games: " + str(available_games) + ". Default is 'Riverraid'.", default="Riverraid")
        parser.add_argument("-m", "--mode", help="Choose from available modes: ddqn_train, ddqn_test. Default is 'ddqn_train'.", default="ddqn_train")
        parser.add_argument("-n", "--net", help="Network name to use for testing. Default is 'model'", default="model")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 0 - do not render, 1 - normal, 2 - network vision, 3 - gif file + normal, 4 - featuremap", default=0, type=int)
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit", help="Choose after how many runs we should stop. Default is None (no limit).", default=None, type=int)
        parser.add_argument("-c", "--clip", help="Choose whether we should clip rewards to (0, 1) range. Default is 'False'", default=False, type=bool)
        parser.add_argument("-s", "--skip", help="Max random number of frames to skip on reset. Default is 0", default=0, type=int)
        args = parser.parse_args()
        game_mode = args.mode
        model_name = args.net
        game_name = args.game
        render_mode = args.render
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        clip = args.clip
        skip = args.skip
        print ("Selected game: " + str(game_name))
        print ("Selected mode: " + str(game_mode))
        print ("Model name: " + str(model_name))
        print ("Render mode: " + str(render_mode))
        print ("Should clip: " + str(clip))
        print ("Should skip frames: " + str(skip))
        print ("Total step limit: " + str(total_step_limit))
        print ("Total run limit: " + str(total_run_limit))
        return game_name, game_mode, model_name, render_mode, total_step_limit, total_run_limit, clip, skip

    def _game_model(self, game_mode, model_name, game_name, action_space):
        if game_mode == "ddqn_train":
            return DDQNTrainer(game_name, INPUT_SHAPE, action_space, model_name)
        elif game_mode == "ddqn_test":
            return DDQNSolver(game_name, INPUT_SHAPE, action_space, model_name)
        else:
            print ("Unrecognized mode. Use --help")
            exit(1)

if __name__ == "__main__":
    Main()
