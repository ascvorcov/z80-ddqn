import numpy as np
import os
import random
import shutil
import gzip
import struct
import time
import cv2

from statistics import mean
from base_game_model import BaseGameModel
from convolutional_neural_network import ConvolutionalNeuralNetwork
from frame import Frame
from memory import Memory

GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

class DDQNGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path, train_data_path=None):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)

        self.model_path = model_path
        self.train_data_path = train_data_path
        self.ddqn = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        self.ddqn.summary()
        if os.path.isfile(self.model_path):
            self.ddqn.load_weights(self.model_path)
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))

    def _save_model(self):
        self.ddqn.save_weights(self.model_path)


class DDQNSolver(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space, model_name):
        logging_path       = "./output/" + game_name + "/testing/log/" + self._get_date() + "/"
        testing_model_path = "./output/" + game_name + "/testing/" + model_name + ".h5"
        assert os.path.exists(testing_model_path), "No testing model found: " + str(testing_model_path)
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN testing",
                               input_shape,
                               action_space,
                               logging_path,
                               testing_model_path)

    def move(self, state):
        time.sleep(0.1)
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        x = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
        q_values = self.ddqn.predict(x, batch_size=1)

        return np.argmax(q_values[0])


class DDQNTrainer(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space, model_name):
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN training",
                               input_shape,
                               action_space,
                               "./output/" + game_name + "/training/log/" + self._get_date() + "/",
                               "./output/" + game_name + "/training/" + model_name + ".h5",
                               "./output/" + game_name + "/training/training_data.gz")

        self.ddqn_target = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        self._reset_target_network()
        self._load_training_data()

    def _load_training_data(self):
        self.epsilon = EXPLORATION_MAX
        self.memory = Memory(MEMORY_SIZE)
        self.struct_header = struct.Struct("IIIIId")
        self.struct_record = struct.Struct("IIIBB")
        if not os.path.isfile(self.train_data_path):
            return
        print("found training data in %s, loading" % (self.train_data_path,))
        with gzip.open(self.train_data_path, "rb") as f:
            hdr = f.read(self.struct_header.size)
            frame_count, mem_count, img_size, ir, its, eps = self.struct_header.unpack_from(hdr)
            self.initial_run = ir
            self.initial_total_step = its
            self.epsilon = eps
            
            # load frames into simple list
            all_frames = []
            for i in range(frame_count):
                if i % 1000 == 0:
                    print("loaded %d frames out of %d \r" % (i, frame_count), end = "")
                data = bytearray(f.read(img_size))
                all_frames.append(Frame([data]))
            print("loaded %d frames" % (frame_count,))
            
            # load records into memory, reference frames in list by index
            recsz = self.struct_record.size

            def _load_record(f):
                if self.memory.size() % 1000 == 0:
                    print("loaded %d records out of %d \r" % (i, mem_count), end = "")
                ics,ins,rw,ac,t = self.struct_record.unpack_from(f.read(recsz))
                cs = all_frames[ics]
                ns = all_frames[ins]
                data = {"current_state": cs,
                        "action": ac,
                        "reward": rw,
                        "next_state": ns,
                        "terminal": True if t == 1 else False}
                return data

            self.memory.load(f, _load_record)
            print("loaded %d records" % (mem_count,))
            all_frames = None

    def save(self, run, total_step):
        if not os.path.exists(os.path.dirname(self.train_data_path)):
            os.makedirs(os.path.dirname(self.train_data_path))
        if os.path.isfile(self.train_data_path):
            if os.path.isfile(self.train_data_path + ".bak"):
                os.remove(self.train_data_path + ".bak")
            os.rename(self.train_data_path, self.train_data_path + ".bak")

        with gzip.open(self.train_data_path, "wb") as f:
            all_frames = {}
            mem_count = self.memory.size()
        
            # first count and index all unique frames
            print("indexing frame data")
            for i in range(mem_count): 
                r = self.memory.get_data(i)
                cs = r["current_state"]
                ns = r["next_state"]
                if cs not in all_frames: all_frames[cs] = cs.index = len(all_frames)
                if ns not in all_frames: all_frames[ns] = ns.index = len(all_frames)
        
            # save all frames first, ordered by index
            print("frame data indexing complete, saving frames")
            frame_count = len(all_frames)
            img_size = len(next(iter(all_frames)).as_bytes())
            f.write(self.struct_header.pack(frame_count,mem_count,img_size,run,total_step,self.epsilon))
            processed_records = 0
            for k in sorted(all_frames,key=lambda x: x.index):
                if processed_records%1000==0:
                    print("saved %d records out of %d \r" % (processed_records, frame_count), end = "")
                f.write(k.as_bytes()) 
                processed_records = processed_records+1

            def _save_record(f, record):
                cs = record["current_state"]
                ns = record["next_state"]
                rw = record["reward"]
                ac = record["action"]
                t = 1 if record["terminal"] else 0
                f.write(self.struct_record.pack(cs.index,ns.index,rw,ac,t))
        
            # now save all memory records with frame index instead of real frame
            print("saved %d frames, now saving %d transitions" % (frame_count, mem_count))
            self.memory.save(f, _save_record)
            print("saved %d transitions" % (mem_count,))
            print("releasing memory (can take several minutes)...");

    def move(self, state):
        if np.random.rand() < self.epsilon or self.memory.size() < REPLAY_START_SIZE:
            return random.randrange(self.action_space)
        q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])

    def remember(self, current_state, action, reward, next_state, terminal):
        data = {"current_state": current_state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "terminal": terminal}
        self.memory.append(data)

    def step_update(self, total_step):
        if self.memory.size() < REPLAY_START_SIZE:
            return

        if total_step % TRAINING_FREQUENCY == 0:
            loss, accuracy, average_max_q = self._train()
            self.logger.add_loss(loss)
            self.logger.add_accuracy(accuracy)
            self.logger.add_q(average_max_q)

        self._update_epsilon()

        if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            self._save_model()

        if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self._reset_target_network()
            print("{{'metric': 'epsilon', 'value': {}}}".format(self.epsilon))
            print("{{'metric': 'total_step', 'value': {}}}".format(total_step))

    def _train(self):
        batch = self.memory.sample(BATCH_SIZE)
        if len(batch) < BATCH_SIZE:
            return

        current_states = []
        q_values = []
        max_q_values = []
        sample_weights = []
        for idx, entry, weight in batch:
            sample_weights.append(weight)
            current_state = np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0)
            next_state    = np.expand_dims(np.asarray(entry["next_state"]   ).astype(np.float64), axis=0)
            current_states.append(current_state)
            next_state_prediction = self.ddqn_target.predict(next_state).ravel()
            self_predict          = self.ddqn.predict(current_state);

            next_q_value = np.max(next_state_prediction)
            q = list(self_predict[0])

            reward = entry["reward"]
            action = entry["action"]
            isTerminal = entry["terminal"]

            discountedReward = 0
            if isTerminal:
                discountedReward = reward
            else:
                discountedReward = reward + GAMMA * next_q_value

            err = abs(q[action] - discountedReward)
            q[action] = discountedReward

            q_values.append(q)
            max_q_values.append(np.max(q))
            self.memory.update(idx, err)

        fit = self.ddqn.fit(np.asarray(current_states).squeeze(),
                            np.asarray(q_values).squeeze(),
                            sample_weight = np.asarray(sample_weights),
                            batch_size = BATCH_SIZE,
                            verbose = 0)
        loss = fit.history["loss"][0]
        accuracy = fit.history["accuracy"][0]

        return loss, accuracy, mean(max_q_values)

    def _update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def _reset_target_network(self):
        self.ddqn_target.set_weights(self.ddqn.get_weights())