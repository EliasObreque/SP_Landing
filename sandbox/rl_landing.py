# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:19:01 2023

@author: FATE
"""

import os

import matplotlib.pyplot as plt

from CustomEnvironment import CustomEnv

from stable_baselines3 import PPO # Algorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#%%
my_env = CustomEnv()

print(my_env.reset())
#%%

episodes = 5

for episode in range(1, episodes + 1):
    state = my_env.reset()
    print(state)
    terminated = False
    score = 0
    while not terminated:
        my_env.render()
        action = my_env.action_space.sample()
        n_state, reward, terminated, info = my_env.step(action)
        score += reward
    print("Epidode: {} Score: {}".format(episode, score))
    my_env.plotter()

#%%
log_path = os.path.join('Training', 'Logs')
model = PPO('MlpPolicy', my_env, verbose=1, tensorboard_log=log_path, learning_rate=1e-2)

#%%
model.learn(total_timesteps=50000, progress_bar=True)

#%%
shower_path = os.path.join('Training', 'Saved Models', 'CustomEnvTest')
model.save(shower_path)

# del model
# model = PPO.load(shower_path, my_env)

#%%


episodes = 10
for episode in range(1, episodes + 1):
    state = my_env.reset()
    terminated = False
    score = 0
    while not terminated:
        my_env.render()
        action = my_env.action_space.sample()
        n_state, reward, terminated, info = my_env.step(action)
        score += reward
    print("Epidode: {} Score: {}".format(episode, score))
    my_env.plotter()


a = evaluate_policy(model, my_env, n_eval_episodes=10, render=True)

print(a)
