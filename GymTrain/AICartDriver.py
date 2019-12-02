import gym
import numpy as np
import random
# import tensorflow as tf


env = gym.make('CartPole-v1')
env.reset()
for _ in range(1000):
	env.render()
	# next_action = env.action_space.sample()
	# print(next_action)
	env.step(env.action_space.sample())

env.close()