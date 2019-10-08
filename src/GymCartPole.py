# GymCartPole.py
# Author: Grzegorz Krug

# Pong-v0  /v4
# CartPole-v0

import gym

env = gym.make('CartPole-v0')

for i_episode in range(50):
    observation = env.reset()
    for t in range(2000):
        env.render()
##        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
