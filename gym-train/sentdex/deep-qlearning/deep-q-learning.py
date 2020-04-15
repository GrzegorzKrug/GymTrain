from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque

import tensorflow as tf
import numpy as np
import time


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=config)

REPLAY_MEMORY_SIZE = 50_000
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
    def __init__(self):

        # Main Model, we train it every step
        self.model = self.create_model()

        # Target model this is what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

    @staticmethod
    def create_model():
        model = Sequential([
                Conv2D(256, (5, 5), input_shape=env.OBSERVATION_SPACE_VALUES, activation='relu'),
                MaxPooling2D(),
                Dropout(0.2),

                Conv2D(256, (5, 5), activation='relu'),
                MaxPooling2D(),
                Dropout(0.2),

                Flatten(),

                Dense(64, activation='relu'),
                Dense(env.ACTION_SPACE_SIZE, activation='linear')
        ])
        model.compile(optimizer=Adam(lr=0.001),
                      loss="mse",
                      metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_sq(self, state, step):
        sq = self.model.predict(
                np.array(state).reshape(-1, *state.shape)/255[0]
        )
        return sq

    def train(self, terminal_state, step):


# with tf.compat.v1.Session(config=config) as sess:
#     time.sleep(5)


print("Session closed.")
