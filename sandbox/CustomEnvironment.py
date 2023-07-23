# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:16:59 2023

@author: FATE
"""

from gym import Env
from gym.spaces import Discrete, Box

from core.dynamics.OneDCoordinate import LinearCoordinate

import numpy as np
import random
import matplotlib.pyplot as plt


#%%


class CustomEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([2000]))
        self.__state = 2000
        self.state = 2000 #+ np.random.normal(0, scale=[50, 5, 0.1])
        self.shower_leangth_init = 1e4
        self.shower_leangth = self.shower_leangth_init

        self.hist_step = []
        self.hist_state = []
        self.hist_thrust = []
        self.dt = 1
        self.k  = 1
        self.thrust_on = 0
        self.burn_action = 500 / self.dt  # step
        self.dynamics = LinearCoordinate(self.dt, -1.67, 24)

    # def step(self, action):
    #     thrust = 0.0
    #     reward = 0
    #     if action == 1 and self.thrust_on <= self.burn_action or (0 < self.thrust_on <= self.burn_action):
    #         self.thrust_on += 1
    #         thrust = 100
    #     if self.thrust_on == 1:
    #         reward -= int(self.__state[0] * 0.01)
    #         self.k = 0
    #     m_dot = thrust / (240 * 9.8)
    #     # reward -= int(abs(self.__state[1]) * 0.001)
    #     self.__state = self.dynamics.rungeonestep(self.__state, thrust, m_dot)
    #
    #     self.shower_leangth -= 1
    #     if self.shower_leangth <= 0 or self.__state[0] < 0 or self.__state[2] < 1:
    #         done = True
    #     else:
    #         done = False
    #     info = {}
    #     if done:
    #         reward += int(1000 - abs(self.state[1]))
    #
    #     self.hist_step.append(self.hist_step[-1] + 1)
    #     self.hist_state.append(self.__state)
    #     self.hist_thrust.append(thrust)
    #     self.state = self.__state[:2]
    #     return self.state, reward, done, info

    def step(self, action):
        if abs(1800 - self.state) < 200:
            reward = 50
        elif abs(1800 - self.state) < 100:
            reward = 100
        elif abs(1800 - self.state) < 50:
            reward = 200
        else:
            reward = -10
        if action == 0:
            level = 0
        elif action == 1:
            level = -1
        elif action == 2:
            level = 1
        else:
            level = 0
        self.state += 10 * level
        done = False
        self.shower_leangth -= 1
        if self.shower_leangth <= 0 or self.state <0:
            done = True
        info = {}
        self.hist_step.append(self.hist_step[-1] + 1)
        self.__state = self.state
        self.hist_state.append(self.__state)
        self.hist_thrust.append(10 * level)
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        pass
    
    def reset(self):
        self.__state = 2000
        self.state = 2000
        self.shower_leangth = self.shower_leangth_init
        self.hist_step = []
        self.hist_state = []
        self.hist_thrust = []
        self.k = 1
        self.thrust_on = 0
        self.hist_step.append(0)
        self.hist_state.append(self.__state)
        self.hist_thrust.append(0)
        return self.state

    def plotter(self):
        plt.figure()
        plt.plot(np.array(self.hist_step) * self.dt, np.array(self.hist_state))
        plt.grid()
        plt.figure()
        plt.plot(np.array(self.hist_step) * self.dt, np.array(self.hist_thrust))
        plt.grid()
        plt.show()


if __name__ == '__main__':
    env = CustomEnv()
    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        terminated = False
        score = 0
        while not terminated:
            env.render()
            action = random.choice([0, 1])
            n_state, reward, terminated, info = env.step(action)
            score += reward
        print("Epidode: {} Score: {}".format(episode, score))