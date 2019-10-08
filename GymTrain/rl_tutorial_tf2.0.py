# rl_tutorial.tf2.0.py
#
# deep reinforcement learning (DRL)
"""TensorFlow's eager execution is an imperative programming environment
that evaluates operations immediately, without building graphs: operations
return concrete values instead of constructing a computational graph to run later.
This makes it easy to get started with TensorFlow and debug models
"""
import tensorflow as tf
from tensorflow import keras
import random
import gym
import numpy as np
import math
import matplotlib.pyplot as plt


# Checking TF version, its ok
tf.__version__  # '2.0.0-rc0'
tf.executing_eagerly()  # True


# Note that we’re now in eager mode by default!
print("1 + 2 + 3 + 4 + 5 =", tf.reduce_sum([1, 2, 3, 4, 5]))
# 1 + 2 + 3 + 4 + 5 = tf.Tensor(15, shape=(), dtype=int32)


# Deep Actor-Critic Methods
#
# While much of the fundamental RL theory was developed on the tabular cases,
# modern RL is almost exclusively done with function approximators,
# such as artificial neural networks. Specifically, an RL algorithm is considered “deep”
# if the policy and value functions are approximated with deep neural networks.


# (Asynchronous) Advantage Actor-Critic
#
# Over the years, a number of improvements have been added to address sample efficiency
# and stability of the learning process.
#
# First, gradients are weighted with returns: discounted future rewards,
# which somewhat alleviates the credit assignment problem, and resolves theoretical issues
# with infinite timesteps.
#
# Second, an advantage function is used instead of raw returns.
# Advantage is formed as the difference between returns
# and some baseline (e.g. state-action estimate)
# and can be thought of as a measure of how good a given action is compared to some average.
#
# Third, an additional entropy maximization term is used in objective
# function to ensure agent sufficiently explores various policies.
# In essence, entropy measures how random a probability distribution is,
# maximized with uniform distribution.
#
# Finally, multiple workers are used in parallel to speed up sample gathering
# while helping decorrelate them during training.


# Incorporating all of these changes with deep neural networks
# we arrive at the two of the most popular modern algorithms:
# (asynchronous) advantage actor critic,
# or A3C/A2C for short.
# The difference between the two is more technical than theoretical:
# as the name suggests, it boils down to how the parallel workers estimate their gradients
# and propagate them to the model.




