from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque

import tensorflow as tf
import numpy as np
import time
import datetime
import random

import gym

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)

REPLAY_MEMORY_SIZE = 20_000
MIN_REPLAY_MEMORY_SIZE = 5_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5

EPOCHS = 10_000
EPS_END_AT = int(EPOCHS * 0.90)
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
        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.name = f"logs/{MODEL_NAME}-{dt.tm_mon}{dt.tm_mday}--{dt.tm_hour}{dt.tm_min}{dt.tm_sec}"
        print(self.name)

        # Target model this is what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # self.modifier_tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.tensorboard = TensorBoard(log_dir=self.name)
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

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # mini batch:
        # (array([-0.49398561,  0.00714135]), 1, array([-0.48706607,  0.00691954]), -1.0, False)
        minibatch = random.sample(self.replay_memory, self.mini_batch_size)
        old_states = np.array([transition[0] for transition in minibatch])
        new_states = np.array([transition[2] for transition in minibatch])

        current_qs_list = self.model.predict(old_states)
        future_qs_list = self.target_model.predict(new_states)

        X = []
        y = []

        for index, (old_state, action, new_state, reward, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            old_qs = current_qs_list[index]
            old_qs[action] = new_q

            X.append(old_state)
            y.append(old_qs)

        X = np.array(X)
        y = np.array(y)

        self.model.fit(
                X, y,
                batch_size=self.mini_batch_size,
                verbose=0, shuffle=False,
                callbacks=[self.tensorboard]
        )

        if terminal_state:
            self.target_update_counter += 1

        # if self.target_update_counter > UPDATE_TARGET_EVERY:
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0


env = gym.make("MountainCar-v0")
action_space = 3
obs_space = (2, )

agent = DQNAgent(
        observation_space_vals=obs_space,
        action_space_size=action_space
)


eps_iter = iter(np.linspace(0.8, 0.2, EPS_END_AT))

for epoch in range(EPOCHS):
    done = False
    cost = 0
    try:
        eps = next(eps_iter)
    except StopIteration:
        eps = 0
    old_state = env.reset()

    if not epoch % 100:
        agent.train(True)

    while not done:

        # Preparated Actions
        # if not x % interval:
        #     flag ^= True
        #     interval += interval + 7 + interval // 7
        #     print("Flip", f"next interval: {interval}")
        # action = 0 if flag else 2
        if np.random.random() < eps:
            action = np.random.randint(0, action_space)
        else:
            _pred = np.array(old_state).reshape(1, -1)
            predictions = agent.model.predict(_pred)
            action = np.argmax(predictions)

        new_state, reward, done, _ = env.step(action)
        new_transition = (old_state, action, new_state, reward, done)
        agent.update_replay_memory(new_transition)

        cost += reward
        # env.render()
        # time.sleep(0.005)

        if done:
            break

        old_state = new_state
    print(f"Epoch {epoch:^3} finished with cost: {cost:^9}, eps: {eps:^5}")

# print(state)  # [Position, velocity]
print("Session closed.")

