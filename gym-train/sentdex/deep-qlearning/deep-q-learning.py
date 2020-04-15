from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque

import tensorflow as tf
import numpy as np
import time
import random

import gym

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=config)

REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5

DISCOUNT = 0.99

MODEL_NAME = "256x256"


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overridden, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overridden
    # We train for one batch only, no need to save anything at batch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overridden, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self, action_space_size, observation_space_vals, mini_batch_size=64):
        self.observation_space_vals = observation_space_vals
        self.action_space_size = action_space_size
        self.mini_batch_size = mini_batch_size

        # Main Model, we train it every step
        self.model = self.create_model()

        # Target model this is what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential([
                Flatten(input_shape=self.observation_space_vals),
                Dense(256, activation='relu'),
                Dropout(0.2),

                Flatten(),

                Dense(64, activation='relu'),
                Dense(self.action_space_size, activation='linear')
        ])
        model.compile(optimizer=Adam(lr=0.001),
                      loss="mse",
                      metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_sq(self, state):
        sq = self.model.predict(
                np.array(state).reshape(-1, *state.shape)/255[0]
        )
        return sq

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.mini_batch_size)
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        X = np.array(X)
        y = np.array(y)

        self.model.fit(
                X, y,
                batch_size=self.mini_batch_size,
                verbose=0, shuffle=False
                # callbacks=[self.tensorboard] if terminal_state else None
        )

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


env = gym.make("MountainCar-v0")
state = env.reset()
action_space = 3
obs_space = (2, 1)
agent = DQNAgent(
        observation_space_vals=obs_space,
        action_space_size=action_space
)

flag = True
interval = 10
cost = 0
for x in range(300):

    if not x % interval:
        flag ^= True
        interval += interval + 3 + interval // 7
        print("Flip", f"next interval: {interval}")
    action = 0 if flag else 2

    state, reward, done, _ = env.step(action)

    # print(state)  # [Position, velocity]
    # print("reward:", reward)
    cost += reward
    env.render()
    time.sleep(0.005)

    if done:
        break
print(f"Total cost: {cost}")
# with tf.compat.v1.Session(config=config) as sess:
#     time.sleep(5)


print("Session closed.")
