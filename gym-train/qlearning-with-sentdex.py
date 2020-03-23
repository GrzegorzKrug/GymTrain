import gym
import time
import numpy as np

env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNT = 0.95  # weight, how important are future action over current
EPISODES = 10000

SHOW_EVERY = 1000

DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

eps = 0.8  # not a constant, going to be decayed
START_EPSILON_DECAYING = 10
END_EPSILON_DECAYING = EPISODES // 2
# END_EPSILON_DECAYING = 100


q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    dc_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(dc_state.astype(np.int))


class EpsIterator:
    def __init__(self, start, stop, n):
        self.now = start
        self.start = start
        self.stop = stop
        self.n = n

    def __next__(self):
        for cur_eps in np.linspace(self.start, self.stop, self.n):
            self.now = cur_eps
            print('Next')
            yield self.now

    def __iter__(self, *args):
        print(f'Iter, args: {args}')
        print(f"Now: {self.now}")
        return self


def eps_function(start, stop, n):
    for this_eps in np.linspace(start, stop, n):
        yield this_eps


# eps_iterator = iter(EpsIterator(epsilon, 0, END_EPSILON_DECAYING - START_EPSILON_DECAYING))
eps_iterator = iter(np.linspace(eps, 0, END_EPSILON_DECAYING - START_EPSILON_DECAYING))


for episode in range(EPISODES):
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        try:
            eps = next(eps_iterator)
        except StopIteration:
            eps = 0

    print(f"Episode: {episode}, Epsilon: {eps}")
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > eps:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()
            time.sleep(0.007)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])  # highest value of q action
            current_q = q_table[discrete_state + (action, )]  # current q + action, q before movement
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"We reached goal at: {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

        # if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        #     epsilon = epsilon + epsilon_decay_value

    env.close()


