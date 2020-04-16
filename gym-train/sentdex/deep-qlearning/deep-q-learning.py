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
import matplotlib.pyplot as plt
import gym
import os


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)

SIM_COUNT = 20

REPLAY_MEMORY_SIZE = 5 * SIM_COUNT * 200
MIN_REPLAY_MEMORY_SIZE = 2 * SIM_COUNT * 200
MINIBATCH_SIZE = 512
DISCOUNT = 0.99

EPOCHS = 200
INITIAL_EPS = 0.6
END_EPS = 0.1
EPS_END_AT = 50


MODEL_NAME = "Relu32-Relu32-LinOut-"
SHOW_EVERY = EPOCHS // 4
TRAIN_EVERY = 5
SHOW_LAST = False


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
        self.name = f"{MODEL_NAME}--{dt.tm_mon}-{dt.tm_mday}--{dt.tm_hour}-{dt.tm_min}-{dt.tm_sec}"

        # Target model this is what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # self.modifier_tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.tensorboard = TensorBoard(log_dir=self.name)
        self.target_update_counter = 0

        self.load_model()

    def create_model(self):
        model = Sequential([
                Flatten(input_shape=self.observation_space_vals),
                Dense(32, activation='relu'),
                Dense(32, activation='relu'),
                # Dropout(0.2),

                # Flatten(),

                Dense(self.action_space_size, activation='linear')
        ])
        model.compile(optimizer=Adam(lr=0.01),
                      loss="mse",
                      metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def save_model(self):
        self.model.save_weights("models/last_model", overwrite=True)

    def load_model(self):
        if os.path.isfile("models/last_model.index"):
            print("Loading model")
            self.model.load_weights("models/last_model")
            self.target_model.load_weights("models/last_model")

    def get_sq(self, state):
        sq = self.model.predict(
                np.array(state).reshape(-1, *state.shape)[0]
        )
        return sq

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # mini batch:
        # (array([-0.49398561,  0.00714135]), 1, array([-0.48706607,  0.00691954]), -1.0, False)
        memory = random.sample(self.replay_memory, self.mini_batch_size)
        old_states = np.array([transition[0] for transition in memory])
        new_states = np.array([transition[2] for transition in memory])

        current_qs_list = self.model.predict(old_states)
        future_qs_list = self.target_model.predict(new_states)

        X = []
        y = []

        for index, (old_state, action, new_state, reward, done) in enumerate(memory):
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
        history = self.model.fit(
                X, y,
                verbose=0, shuffle=False, epochs=1,
                batch_size=64
                # callbacks=[self.tensorboard]
        )
        print(f"Train done. Loss: {history.history['loss'][-1]:>2.4f}, Accuracy: {history.history['accuracy'][-1]:>2.4f}")
        if terminal_state:
            self.target_update_counter += 1

        # if self.target_update_counter > UPDATE_TARGET_EVERY:
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

        self.save_model()


ENVS = []

for x in range(SIM_COUNT):
    ENVS.append(gym.make("MountainCar-v0"))

action_space = 3
obs_space = (2, )

agent = DQNAgent(
        observation_space_vals=obs_space,
        action_space_size=action_space,
        mini_batch_size=MINIBATCH_SIZE
)


eps_iter = iter(np.linspace(INITIAL_EPS, END_EPS, EPS_END_AT))

x_graph = []
y_graph = []
eps_graph = []


for epoch in range(EPOCHS):
    Cost = [0] * SIM_COUNT
    full_cost = 0
    Envs = ENVS.copy()

    New_states = []
    for env in Envs:
        New_states.append(env.reset())
    New_states = np.array(New_states)

    if not epoch % TRAIN_EVERY:
        agent.train(True)

    if not epoch % SHOW_EVERY:
        render = True
        eps = 0
    else:
        render = False
        try:
            eps = next(eps_iter)
        except StopIteration:
            eps_iter = iter(np.linspace(INITIAL_EPS, END_EPS, EPS_END_AT))
            eps = 0
    if epoch == EPOCHS - 1 and SHOW_LAST:
        render = True
        eps = 0
        input("Last agent...")

    while True:
        # Preparated Actions
        # if not x % interval:
        #     flag ^= True
        #     interval += interval + 7 + interval // 7
        #     print("Flip", f"next interval: {interval}")
        # action = 0 if flag else 2
        Done = [False] * len(Envs)

        Old_states = np.array(New_states)

        New_states = []

        if np.random.random() < eps:
            Actions = [np.random.randint(0, action_space)] * len(Envs)
        else:
            Old_states = np.array(Old_states).reshape(-1, 2)
            Predictions = agent.model.predict(Old_states)
            Actions = np.argmax(Predictions, axis=1)

        for index, env in enumerate(Envs):
            new_state, reward, done, _ = env.step(Actions[index])
            if abs(new_state[1]) > 0.001:
                reward += 0.2

            if abs(new_state[1]) > 0.003:
                reward += 0.4

            reward -= 0.2
            reward -= 0.4

            # else:
            #     print(new_state)

            new_transition = (Old_states[index], Actions[index], new_state, reward, done)
            agent.update_replay_memory(new_transition)

            Cost[index] += reward
            Done[index] = done
            New_states.append(new_state)

            if index == 0 and render:
                env.render()
                time.sleep(0.001)

        for ind_d in range(len(Envs)-1, -1, -1):
            # print(ind_d)
            if Done[ind_d]:
                if ind_d == 0:
                    Envs[0].close()
                full_cost += Cost[ind_d]
                x_graph.append(epoch)
                y_graph.append(Cost[ind_d])
                Cost.pop(ind_d)
                Envs.pop(ind_d)
                New_states.pop(ind_d)

        if len(Envs) <= 0:
            break

    eps_graph.append(eps)

    print(f"Epoch {epoch:^3} finished with cost_avg: {full_cost/SIM_COUNT:>3.2f}, eps: {eps:^5.3f}")
    print(" = ="*10)

plt.figure(figsize=(16, 9))
plt.subplot(211)
plt.scatter(x_graph, y_graph, marker='s', alpha=0.3, edgecolors='m', label="Cost")

plt.subplot(212)
plt.plot(eps_graph, label="Epsilon")
plt.xlabel("Epochs")

plt.suptitle(f"{agent.name}")
plt.savefig(f"{agent.name}.png")
plt.show()
print("Session closed.")

