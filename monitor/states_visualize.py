"""
Created by:

@author: Elias Obreque
@Date: 3/14/2021 1:49 PM 
els.obrq@gmail.com

"""
import json

import matplotlib.pyplot as plt
import numpy as np

from tools.Viewer import *


def get_state(folder_file_name, engine=None):
    with open(folder_file_name) as file:
        data = json.load(file)

    n_case = data["N_case"]
    result_ne = data[str(ne)]

    pos = []
    vel = []
    mass = []
    for i in range(n_case):
        mass.append(result_ne['Case'+str(i)]['response']['mass[kg]'][-1])
        index_pos_ = np.argmin(np.abs(result_ne['Case'+str(i)]['response']['Pos[m]']))
        pos.append(result_ne['Case'+str(i)]['response']['Pos[m]'][index_pos_])
        vel.append(result_ne['Case'+str(i)]['response']['Pos[m]'][index_pos_])
    return pos, vel, mass


# Alt-noise
folder_name = "logs/Only_GA_all/"
folder_name = "logs/Only_GA_no_noise/"

# folder_name += "regressive/2022-02-20T14-41-53/"
folder_name += "regressive/2022-03-11T13-30-34/"
file_name = "eva_reg_performance_data.json"
ne = 10
pos, vel, mass = get_state(folder_name + file_name, engine=ne)

print(24 - min(mass), 24 - np.mean(mass), np.std(mass - np.mean(mass)))


def plot_distribution(pos_, vel_, mass_, folder_name=None, file_name=None, save=False):
    fig_dist, axs_dist = plt.subplots(1, 3)
    axs_dist[0].set_xlabel('Altitude [m]')
    axs_dist[1].set_xlabel('Velocity [m/s]')
    axs_dist[2].set_xlabel('Landing mass [kg]')
    final_pos = pos_
    final_vel = vel_
    w = 4
    n = int(len(final_pos) / w)
    if n == 0:
        n = 1
    axs_dist[0].hist(final_pos, bins=n, color='#0D3592', rwidth=0.85)
    axs_dist[0].axvline(np.mean(final_pos), color='k', linestyle='dashed', linewidth=0.8)
    axs_dist[0].grid()
    axs_dist[1].hist(final_vel, bins=n, color='#0D3592', rwidth=0.85)
    axs_dist[1].axvline(np.mean(final_vel), color='k', linestyle='dashed', linewidth=0.8)
    axs_dist[1].grid()
    axs_dist[2].hist(mass, bins=n, color='#0D3592', rwidth=0.85)
    axs_dist[2].axvline(np.mean(mass), color='k', linestyle='dashed', linewidth=0.8)
    axs_dist[2].grid()
    if save:
        if os.path.isdir("./logs/" + folder_name) is False:
            temp_list = folder_name.split("/")
            fname = ''
            for i in range(len(temp_list) - 1):
                fname += temp_list[:i + 1][i]
                if os.path.isdir("./logs/" + fname) is False:
                    os.mkdir("./logs/" + fname)
                fname += "/"
        fig_dist.savefig("./logs/" + folder_name + file_name + '.png', dpi=300, bbox_inches='tight')
        fig_dist.savefig("./logs/" + folder_name + file_name + '.eps', format='eps', bbox_inches='tight')
    return [np.mean(final_pos), np.mean(final_vel), np.mean(mass_), np.std(final_pos), np.std(final_vel), np.std(mass_)]


plot_distribution(pos, vel, mass)

plt.show()

