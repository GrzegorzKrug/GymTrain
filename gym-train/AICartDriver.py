import gym
import numpy as np
import random
import time

env = gym.make('CartPole-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        time.sleep(0.01)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
