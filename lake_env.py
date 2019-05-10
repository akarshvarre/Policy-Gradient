# coding: utf-8
"""Defines some frozen lake maps."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP
from gym.envs.toy_text import frozen_lake, discrete

from gym.envs.registration import register

import gym
from tester import Tester
import tester
import numpy as np
'''
action_names = {LEFT: 'LEFT', RIGHT: 'RIGHT', DOWN: 'DOWN', UP: 'UP'}

register(
    id='Stochastic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': True})

env = gym.make('Stochastic-4x4-FrozenLake-v0', is_slippery = True)
env.reset()
'''
t = Tester()

gamma = 0.9

#policy = np.random.randint(4, size=16)
'''
policy = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]).T

opt_policy,value,i,it = t.policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3)
#value = t.evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3)
print(opt_policy)
print(value)
'''


env = gym.make('CartPole-v0')
state, ep_reward = env.reset(), 0

action  = t.policy_gradient_test(state)
print(action)