"""
Created by:

@author: Elias Obreque
@Date: 3/14/2021 1:49 PM 
els.obrq@gmail.com

"""
import json

import matplotlib.pyplot as plt

from tools.Viewer import *


def get_performance(folder_file_name):
    with open(folder_file_name) as file:
        data = json.load(file)

    name_thrusters = []
    temp = data
    performance_data = [temp['mean_pos'], temp['mean_vel'], temp['std_pos'], temp['std_vel']]
    n_thrusters = len(performance_data[0])
    name_thrusters.append(int(2))
    return performance_data, n_thrusters, name_thrusters


# Alt-noise
folder_name = "logs/Only_GA_all/regressive/"
folder_name += "2022-02-20T14-41-53/"
file_name = "eva_reg_performance_data.json"
performance_data_regressive_alt, n_thrusters_regressive_alt, _ = get_performance(folder_name + file_name)

folder_name = "logs/Only_GA_all/progressive/"
folder_name += "2022-02-20T14-48-33/"
file_name = "eva_pro_performance_data.json"
performance_data_progressive_alt, n_thrusters_progressive_alt, _ = get_performance(folder_name + file_name)

folder_name = "logs/Only_GA_all/neutral/"
folder_name += "2022-02-20T14-48-52/"
file_name = "eva_neu_performance_data.json"
performance_data_constant_alt, n_thrusters_constant_alt, _ = get_performance(folder_name + file_name)


def plot_errorbar_performance(n_thrusters_constant_, performance_data_constant_,
                              n_thrusters_progressive_, performance_data_progressive_,
                              n_thrusters_regressive_, performance_data_regressive_):
    fig_perf, axs_perf = plt.subplots(2, 1, sharex=True)
    axs_perf[0].set_ylabel('Landing position [m]')
    axs_perf[0].grid()
    lw_data = 0.8
    reg_color = 'r'
    pro_color = 'b'
    con_color = 'g'
    axs_perf[0].errorbar(np.arange(1, 1 + n_thrusters_regressive_), np.array(performance_data_regressive_)[0],
                         yerr=np.array(performance_data_regressive_)[2], fmt='-', capsize=5, color=reg_color,
                         ecolor=reg_color, label='regressive', lw=lw_data)
    axs_perf[0].errorbar(np.arange(1, 1 + n_thrusters_progressive_), np.array(performance_data_progressive_)[0],
                         yerr=np.array(performance_data_progressive_)[2], fmt='-', capsize=5, color=pro_color,
                         ecolor=pro_color, label='Progressive', lw=lw_data)
    axs_perf[0].errorbar(np.arange(1, 1 + n_thrusters_constant_), np.array(performance_data_constant_)[0],
                         yerr=np.array(performance_data_constant_)[2], fmt='-', capsize=5, color=con_color,
                         ecolor=con_color, label='Neutral', lw=lw_data)

    axs_perf[1].set_ylabel('Landing velocity [m/s]')
    axs_perf[1].set_xlabel('Number of thrusters')
    axs_perf[1].grid()
    axs_perf[1].errorbar(np.arange(1, 1 + n_thrusters_regressive_), np.array(performance_data_regressive_)[1],
                         yerr=np.array(performance_data_regressive_)[3], fmt='-', capsize=5, ecolor=reg_color,
                         color=reg_color, label='regressive', lw=lw_data)
    axs_perf[1].errorbar(np.arange(1, 1 + n_thrusters_progressive_), np.array(performance_data_progressive_)[1],
                         yerr=np.array(performance_data_progressive_)[3], fmt='-', capsize=5, ecolor=pro_color,
                         color=pro_color, label='Progressive', lw=lw_data)
    axs_perf[1].errorbar(np.arange(1, 1 + n_thrusters_constant_), np.array(performance_data_constant_)[1],
                         yerr=np.array(performance_data_constant_)[3], fmt='-', capsize=5, ecolor=con_color,
                         color=con_color, label='Neutral', lw=lw_data)
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.legend()
    plt.tight_layout()
    return


def plot_velocity_errorbar_performance(n_thrusters_constant_, performance_data_constant_,
                                       n_thrusters_progressive_, performance_data_progressive_,
                                       n_thrusters_regressive_, performance_data_regressive_):
    lw_data = 0.8
    reg_color = 'r'
    pro_color = 'b'
    con_color = 'g'

    fig_perf = plt.figure()
    plt.ylabel('Landing velocity [m/s]')
    plt.xlabel('Number of thrusters')
    plt.grid()
    plt.errorbar(np.arange(1, 1 + n_thrusters_regressive_), np.array(performance_data_regressive_)[1],
                         yerr=np.array(performance_data_regressive_)[3], fmt='-', capsize=5, ecolor=reg_color,
                         color=reg_color, label='regressive', lw=lw_data)
    plt.errorbar(np.arange(1, 1 + n_thrusters_progressive_), np.array(performance_data_progressive_)[1],
                         yerr=np.array(performance_data_progressive_)[3], fmt='-', capsize=5, ecolor=pro_color,
                         color=pro_color, label='Progressive', lw=lw_data)
    plt.errorbar(np.arange(1, 1 + n_thrusters_constant_), np.array(performance_data_constant_)[1],
                         yerr=np.array(performance_data_constant_)[3], fmt='-', capsize=5, ecolor=con_color,
                         color=con_color, label='Neutral', lw=lw_data)
    plt.legend()
    plt.tight_layout()

    fig_perf.savefig("./logs/comparison.png", dpi=300, bbox_inches='tight')
    fig_perf.savefig("./logs/comparison.eps", format='eps', bbox_inches='tight')
    return


def plot_std_performance(n_thrusters_constant_, performance_data_constant_,
                              n_thrusters_progressive_, performance_data_progressive_,
                              n_thrusters_regressive_, performance_data_regressive_):
    lw_data = 0.8
    reg_color = 'r'
    pro_color = 'b'
    con_color = 'g'
    fig_perf_std, axs_perf_std = plt.subplots(2, 1, sharex=True)
    axs_perf_std[0].set_ylabel('Std of Position [m]')
    axs_perf_std[0].grid()
    axs_perf_std[0].plot(np.arange(1, 1 + n_thrusters_regressive_), np.array(performance_data_regressive_)[2],
                         '-o', color=reg_color, lw=lw_data, label='regressive')
    axs_perf_std[0].plot(np.arange(1, 1 + n_thrusters_progressive_), np.array(performance_data_progressive_)[2],
                         '-o', color=pro_color, lw=lw_data,label='Progressive')
    axs_perf_std[0].plot(np.arange(1, 1 + n_thrusters_constant_), np.array(performance_data_constant_)[2],
                         '-o', color=con_color, lw=lw_data,label='Neutral')

    axs_perf_std[1].set_xlabel('Number of thrusters')
    axs_perf_std[1].set_ylabel('std of Velocity [m/s]')
    axs_perf_std[1].plot(np.arange(1, 1 + n_thrusters_regressive_), np.array(performance_data_regressive_)[3],
                         '-o', color=reg_color, lw=lw_data,label='regressive')
    axs_perf_std[1].plot(np.arange(1, 1 + n_thrusters_progressive_), np.array(performance_data_progressive_)[3],
                         '-o', color=pro_color, lw=lw_data,label='Progressive')
    axs_perf_std[1].plot(np.arange(1, 1 + n_thrusters_constant_), np.array(performance_data_constant_)[3],
                         '-o', color=con_color, lw=lw_data,label='Neutral')
    axs_perf_std[1].grid()
    plt.legend()
    plt.tight_layout()


plot_errorbar_performance(n_thrusters_constant_alt, performance_data_constant_alt,
                          n_thrusters_progressive_alt, performance_data_progressive_alt,
                          n_thrusters_regressive_alt, performance_data_regressive_alt)

plot_std_performance(n_thrusters_constant_alt, performance_data_constant_alt,
                     n_thrusters_progressive_alt, performance_data_progressive_alt,
                     n_thrusters_regressive_alt, performance_data_regressive_alt)

plot_velocity_errorbar_performance(n_thrusters_constant_alt, performance_data_constant_alt,
                                   n_thrusters_progressive_alt, performance_data_progressive_alt,
                                   n_thrusters_regressive_alt, performance_data_regressive_alt)

plt.show()

