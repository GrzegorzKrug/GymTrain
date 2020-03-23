import gym
import numpy as np
import random
import time

env = gym.make('MountainCar-v0')
print(f"Actions: {env.action_space.n}")
print(f"Observations: {env.observation_space}")

Q = np.zeros([2, 3])
G = 0
alpha = 0.618

for episode in range(1, 1001):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while not done:
        action = 1
        state2, reward, done, info = env.step(action)  # 2
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])  # 3
        G += reward
        state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))
