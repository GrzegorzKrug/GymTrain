from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
# from

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import random
import time
import gym
import os


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)

EPOCHS = 300
SIM_COUNT = 1
REPLAY_MEMORY_SIZE = 5 * SIM_COUNT * 200
MIN_REPLAY_MEMORY_SIZE = 2 * SIM_COUNT * 200

# LR = 0.05
MINIBATCH_SIZE = 300 * SIM_COUNT

DISCOUNT = 0.95
AGENT_LR = 0.01
# AGENT_LR = 0.0001

MODEL_NAME = "Relu16-Drop0_2-Relu16-LinOut-1e-4-B_120-Speed_reward"
os.makedirs(MODEL_NAME, exist_ok=True)
LOAD = True

STATE_OFFSET = 0
INITIAL_EPS = 0.7
INITIAL_SMALL_EPS = 0.15
END_EPS = 0

EPS_END_AT = EPOCHS // 5

SHOW_EVERY = EPOCHS * 5
if SHOW_EVERY < 25:
    SHOW_EVERY = 25
TRAIN_EVERY = 1
CLONE_EVERY_TRAIN = 3

SHOW_LAST = False
PLOT_ALL_QS = True
MORE_REWARDS = True

# def state_normalize(state, min_list, max_list):
#     # return state
#     state -= min_list
#     state = state / (max_list - min_list)
#     return state


# # Own Tensorboard class
# class ModifiedTensorBoard(TensorBoard):
#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.step = 1
#         self.writer = tf.summary
#
#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass
#
#     # Overridden, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)
#
#     # Overridden
#     # We train for one batch only, no need to save anything at batch end
#     def on_batch_end(self, batch, logs=None):
#         pass
#
#     # Overridden, so won't close writer
#     def on_train_end(self, _):
#         pass
#
#     # Custom method for saving own metrics
#     # Creates writer, writes custom metrics and closes writer
#     def update_stats(self, **stats):
#         self._write_logs(stats, self.step)


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
        self.train_model = self.create_model()
        self.train_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # self.modifier_tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.tensorboard = TensorBoard(log_dir=self.name)
        self.target_update_counter = 0

        self.load_model()

    def create_model(self):
        model = Sequential([
                # Flatten(),
                Dense(16, activation='relu', input_shape=self.observation_space_vals),
                Dropout(0.2),
                Dense(16, activation='relu', ),
                Dense(self.action_space_size, activation='linear')
        ])
        model.compile(optimizer=Adam(lr=AGENT_LR),
                      loss="mse",
                      metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def save_model(self):
        self.model.save_weights(f"{MODEL_NAME}/model", overwrite=True)

    def load_model(self):
        if LOAD and os.path.isfile(f"{MODEL_NAME}/model"):
            print(f"Loading model: {MODEL_NAME}")
            self.model.load_weights(f"{MODEL_NAME}/model")
            self.train_model.load_weights(f"{MODEL_NAME}/model")
        else:
            print(f"New model: {MODEL_NAME}")

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
        future_qs_list = self.model.predict(new_states)

        X = []
        y = []
        diffs = []
        for index, (old_state, action, new_state, reward, done) in enumerate(memory):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            old_qs = current_qs_list[index]
            # new_q = old_qs[action] * (1 - LR) + LR * new_q
            diffs.append(old_qs[action]-new_q)

            old_qs[action] = new_q

            X.append(old_state)
            y.append(old_qs)

        X = np.array(X)
        y = np.array(y)
        history = self.train_model.fit(
                X, y,
                verbose=0, shuffle=False, epochs=1
                # batch_size=64
                # callbacks=[self.tensorboard]
        )

        diff_min = np.min(diffs)
        diff_max = np.max(diffs)
        diff_avg = np.mean(diffs)

        # print(f"Train - Loss: {history.history['loss'][-1]:>2.4f}, Accuracy: {history.history['accuracy'][-1]:>2.4f}, "
        #       f"Q-diff Min: {diff_min:>5.5f}, Avg: {diff_avg:>5.5f}, Max: {diff_max:>5.5f}")
        
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > CLONE_EVERY_TRAIN:
            self.model.set_weights(self.train_model.get_weights())
            self.target_update_counter = 0
            self.save_model()
            # show_model(self.model.get_weights(), model_name=MODEL_NAME)
            # print("Update weights, save model, save picture done.")

        return diff_min, diff_max, diff_avg


ENVS = []

for x in range(SIM_COUNT):
    ENVS.append(gym.make("MountainCar-v0"))

action_space = 3
obs_space = (2, )
OBS_HIGH = ENVS[0].observation_space.high
OBS_LOW = ENVS[0].observation_space.low

agent = DQNAgent(
        observation_space_vals=obs_space,
        action_space_size=action_space,
        mini_batch_size=MINIBATCH_SIZE)
eps_iter = iter(np.linspace(INITIAL_EPS, END_EPS, EPS_END_AT))

x_graph = []
y_graph = []
average = []
eps_graph = []

diff_x = []
diff_min = []
diff_max = []
diff_avg = []
Predicts = [[], [], [], []]

for epoch in range(EPOCHS):
    Cost = [0] * SIM_COUNT
    full_cost = 0
    Envs = ENVS.copy()

    New_states = []
    for env in Envs:
        state = env.reset()
        # state = state_normalize(state, OBS_LOW, OBS_HIGH)
        New_states.append(state)
    New_states = np.array(New_states)

    if not epoch % TRAIN_EVERY:
        pack = agent.train(True)

        if pack:
            dmin, dmax, davg = pack
            diff_x.append(epoch)
            diff_min.append(dmin)
            diff_max.append(dmax)
            diff_avg.append(davg)

    if not epoch % SHOW_EVERY:
        render = True
        if epoch == 0:
            eps = 0
    else:
        render = False
        if epoch < EPOCHS // 4:
            if epoch < EPOCHS / 6:
                eps = 1.0
            else:
                eps = 0.5
        else:
            try:
                eps = next(eps_iter)
            except StopIteration:
                eps_iter = iter(np.linspace(INITIAL_SMALL_EPS, END_EPS, EPS_END_AT))
                eps = next(eps_iter)

    if epoch == EPOCHS - 1:
        render = True
        eps = 0
        if SHOW_LAST:
            input("Last agent...")

    step = 0
    while True:
        step += 1
        # Preparated Actions
        # if not x % interval:
        #     flag ^= True
        #     interval += interval + 7 + interval // 7
        #     print("Flip", f"next interval: {interval}")
        # action = 0 if flag else 2

        # if step == 100:
        #     print("step 100")
        Done = [False] * len(Envs)
        Old_states = np.array(New_states)

        New_states = []

        if np.random.random() <= eps:
            Actions = np.random.randint(0, action_space, len(Envs))
        else:

            Old_states = np.array(Old_states).reshape(-1, 2)
            predictions = agent.model.predict(Old_states)
            if PLOT_ALL_QS or step < 10 and epoch != 0:
                Predicts[0].append(predictions[0][0])
                Predicts[1].append(predictions[0][1])
                Predicts[2].append(predictions[0][2])
                Predicts[3].append(epoch)

            Actions = np.argmax(predictions, axis=1)

        for index, env in enumerate(Envs):
            new_state, reward, done, _ = env.step(Actions[index])
            # new_state = state_normalize(new_state, OBS_LOW, OBS_HIGH)

            if MORE_REWARDS and not done:
                if abs(new_state[1]) > 0.006:
                    # if render:
                    #     print("small speed")
                    reward += 0.2

                if abs(new_state[1]) > 0.015:
                    # if render:
                    #     print("Speed")
                    reward += 0.8

                reward -= 0.2
                reward -= 0.8

            new_transition = (Old_states[index], Actions[index], new_state, reward, done)
            agent.update_replay_memory(new_transition)

            Cost[index] += reward
            Done[index] = done
            New_states.append(new_state)

            if index == 0 and render:
                arrow = "<" if Actions[0] == 0 else "!" if Actions[0] == 1 else ">"
                # print(new_state[1])
                print(arrow, end='')
                env.render()
                # time.sleep(0.021)

        for ind_d in range(len(Envs)-1, -1, -1):
            if Done[ind_d]:
                if ind_d == 0 and render:
                    Envs[0].close()
                    print()
                    render = False
                full_cost += Cost[ind_d]
                if step < 200:
                    print("GOAL REACHED!")
                x_graph.append(epoch)
                y_graph.append(Cost[ind_d])

                Cost.pop(ind_d)
                Envs.pop(ind_d)
                New_states.pop(ind_d)

        # if len(New_states) == 1:
        #     print("its one")
        #
        # elif len(New_states) == 2:
        #     pass

        if len(Envs) <= 0:
            break

    average.append(full_cost/SIM_COUNT)
    eps_graph.append(eps)

    print(f"Epoch {epoch:^3} end. Average: {full_cost/SIM_COUNT:>3.2f}, eps: {eps:^5.3f}")

agent.save_model()

plt.figure(figsize=(16, 9))
plt.subplot(211)
plt.scatter(x_graph, y_graph, marker='s', alpha=0.3, color='m', label="Cost")
plt.plot(average, label="Average", color="r")
plt.legend(loc='best')
plt.grid()

plt.subplot(212)
plt.plot(eps_graph, label="Epsilon")
plt.legend(loc='best')
plt.xlabel("Epoch")
plt.grid()

# plt.subplot(413)
# plt.plot(diff_x, diff_min, label="q-diff Min")
# plt.plot(diff_x, diff_max, label="q-diff Max")
# plt.plot(diff_x, diff_avg, label="q-diff Avg")
# plt.legend(loc='best')
# plt.grid()

plt.suptitle(f"{agent.name}")
plt.subplots_adjust(hspace=0.4)
plt.savefig(f"{MODEL_NAME}/{agent.name}.png")


plt.figure(figsize=(16, 9))

if PLOT_ALL_QS:
    x_pred = range(len(Predicts[0]))
    ax = plt.subplot(311)
    plt.scatter(x_pred, Predicts[0], label="Q-0(green)", c='g', alpha=0.15, s=10, marker='.')
    plt.legend(loc='best')
    plt.grid()

    ax = plt.subplot(312)
    plt.scatter(x_pred, Predicts[1], label="Q-1(red)", c='r', alpha=0.1, s=10, marker='.')
    plt.legend(loc='best')
    plt.grid()

    ax = plt.subplot(313)
    plt.scatter(x_pred, Predicts[2], label="Q-2(blue)", c='b', alpha=0.1, s=10, marker='.')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Sample")
else:
    plt.scatter(Predicts[3], Predicts[0], label="Q-0(green)", c='g', alpha=0.4, s=10, marker='.')
    plt.scatter(Predicts[3], Predicts[1], label="Q-1(red)", c='r', alpha=0.3, s=10, marker='.')
    plt.scatter(Predicts[3], Predicts[2], label="Q-2(blue)", c='b', alpha=0.2, s=10, marker='.')
    plt.xlabel("Epoch")
    plt.legend(loc='best')
    plt.grid()

plt.savefig(f"{MODEL_NAME}/Qs-{agent.name}.png")
plt.show()
print(f"End: {agent.name}.")
