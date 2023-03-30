# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:23:18 2023

@author: FATE
"""

import random
import gym
import os
#%%
env = gym.make('CartPole-v0')

states = env.observation_space.shape[0]
actions = env.action_space.n

print("States: {}, Actions: {}".format(states, actions))

#%%
episodes = 10
for episode  in range(1, episodes + 1):
    state = env.reset()
    terminated = False
    score = 0
    while not terminated:
        env.render()
        action = random.choice([0, 1])
        n_state, reward, terminated, info = env.step(action)
        score += reward
    print("Epidode: {} Score: {}".format(episode, score))

#%%
env.close()
#%%
# save log
log_path = os.path.join('Training', 'Logs')
print(log_path)
from stable_baselines3 import PPO # Algorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#%%
env = gym.make('CartPole-v0')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)


#%%
model.learn(total_timesteps=20000)

#%%

PPO_PATH = os.path.join('Training', 'Saved Models', 'PPO_Model_CartPole')
model.save(PPO_PATH)
del model
#%%

model = PPO.load(PPO_PATH, env=env)

#%%
evaluate_policy(model, env, n_eval_episodes=10, render=True)

#%%
episodes = 10
for episode  in range(1, episodes + 1):
    obs = env.reset()
    terminated = False
    score = 0
    while not terminated:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, info = env.step(action)
        # obs += np.random.normal(0, 0.05, 4)
        score += reward
    print("Epidode: {} Score: {}".format(episode, score))
    
#%%

training_log = os.path.join(log_path, 'PPO_4')

# !tensorboard --logdir=training_log

#%%





