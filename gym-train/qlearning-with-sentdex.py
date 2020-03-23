import gym
import time
import numpy as np

env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNT = 0.95  # weight, how important are future action over current
EPISODES = 25000

SHOW_EVERY = 1000

DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    dc_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(dc_state.astype(np.int))


for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()
            time.sleep(0.007)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])  # highest value of q action
            current_q = q_table[discrete_state + (action, )]  # current q with action
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0] >= env.goal_position:
            # print(f"We reached goal at: {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    env.close()


