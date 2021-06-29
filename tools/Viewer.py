"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
from matplotlib import pyplot as plt
import os
from matplotlib import patches
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 14


def create_plot():
    plt.figure(1)
    plt.ylabel('Radius [m]')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(2)
    plt.ylabel('Radial Velocity [m/s]')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(3)
    plt.ylabel(r'$\theta$ [rad]')
    plt.xlabel('Time [s]')
    plt.grid()

    plt.figure(4)
    plt.ylabel('Angular velocity [rad/s]')
    plt.xlabel('Time [s]]')
    plt.grid()

    plt.figure(5)
    plt.ylabel('Mass [kg]')
    plt.xlabel('Time [s]]')
    plt.grid()

    plt.figure(6)
    plt.ylabel('Thrust (N)')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(7)
    plt.ylabel('Altitude [m]')
    plt.xlabel('Time [s]')
    plt.grid()

    fig = plt.figure(8)
    ax = fig.gca()
    r_moon = 1738e3
    circle1 = plt.Circle((0, 0), r_moon, color='darkgray')
    plt.ylabel('Orbit position Y [m]')
    plt.xlabel('Orbit position X [m]')
    ax.add_artist(circle1)
    plt.grid()

    plt.figure(9)
    plt.ylabel(r'$\phi$ [rad]')
    plt.xlabel('Time [s]')
    plt.grid()


def set_plot(n_figure, x, y, opt1, opt2, max_H=0, max_V=0):
    plt.figure(n_figure)
    plt.plot(x, y, opt1, lw=1)
    plt.plot(x[0], y[0], opt2, markersize=4)


def show_plot():
    plt.show(block=True)


def plot_main_parameters(time_best, best_pos, best_vel, best_mass, best_thrust, ini_best_individuals,
                         end_best_individuals, save=False, folder_name=None, file_name=None):
    fig_best, axs_best = plt.subplots(2, 2, constrained_layout=True)
    sq = 15
    """
    ini_best_individuals: Initial ignition point
    end_best_individuals: End of burning process
    """

    axs_best[0, 0].set_xlabel('Time [s]')
    axs_best[0, 0].set_ylabel('Position [m]')
    for k in range(len(best_pos)):
        axs_best[0, 0].plot(time_best[k], best_pos[k], lw=0.8)
        axs_best[0, 0].scatter(time_best[k][ini_best_individuals[k]],
                               np.array(best_pos[k])[ini_best_individuals[k]],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[0, 0].scatter(time_best[k][end_best_individuals[k]],
                               np.array(best_pos[k])[end_best_individuals[k]],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
    axs_best[0, 0].grid(True)

    axs_best[0, 1].set_xlabel('Time [s]')
    axs_best[0, 1].set_ylabel('Velocity [m/s]')
    for k in range(len(best_pos)):
        axs_best[0, 1].plot(time_best[k], np.array(best_vel[k]), lw=0.8)
        axs_best[0, 1].scatter(time_best[k][ini_best_individuals[k]],
                               np.array(best_vel[k])[ini_best_individuals[k]],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[0, 1].scatter(time_best[k][end_best_individuals[k]],
                               np.array(best_vel[k])[end_best_individuals[k]],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
    axs_best[0, 1].grid(True)

    axs_best[1, 0].set_xlabel('Time [s]')
    axs_best[1, 0].set_ylabel('Mass [kg]')
    for k in range(len(best_pos)):
        axs_best[1, 0].plot(time_best[k], best_mass[k], lw=0.8)
        axs_best[1, 0].scatter(time_best[k][ini_best_individuals[k]],
                               np.array(best_mass[k])[ini_best_individuals[k]],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[1, 0].scatter(time_best[k][end_best_individuals[k]],
                               np.array(best_mass[k])[end_best_individuals[k]],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
    axs_best[1, 0].grid(True)

    axs_best[1, 1].set_xlabel('Time [s]')
    axs_best[1, 1].set_ylabel('Thrust [N]')
    for k in range(len(best_pos)):
        axs_best[1, 1].plot(time_best[k], np.array(best_thrust[k]), lw=0.8)
        axs_best[1, 1].scatter(time_best[k][ini_best_individuals[k]],
                               np.array(best_thrust[k])[ini_best_individuals[k]],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[1, 1].scatter(time_best[k][end_best_individuals[k]],
                               np.array(best_thrust[k])[end_best_individuals[k]],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
    axs_best[1, 1].grid(True)

    if save:
        if os.path.isdir("./logs/" + folder_name) is False:
            temp_list = folder_name.split("/")
            fname = ''
            for i in range(len(temp_list) - 1):
                fname += temp_list[:i + 1][i]
                if os.path.isdir("./logs/" + fname) is False:
                    os.mkdir("./logs/" + fname)
                fname += "/"
        fig_best.savefig("./logs/" + folder_name + file_name + '.png', dpi=300, bbox_inches='tight')
        fig_best.savefig("./logs/" + folder_name + file_name + '.eps', format='eps')
    plt.draw()
    return


def plot_gauss_distribution(pos_, vel_, land_index, folder_name=None, file_name=None, save=False):
    fig_gauss, axs_gauss = plt.subplots(1, 2)
    axs_gauss[0].set_xlabel('Altitude [m]')
    axs_gauss[1].set_xlabel('Velocity [m/s]')
    final_pos = [pos_[k][land_index[k]] for k in range(len(pos_))]
    final_vel = [vel_[k][land_index[k]] for k in range(len(pos_))]
    w = 4
    n = int(len(final_pos) / w)
    if n == 0:
        n = 1
    axs_gauss[0].hist(final_pos, bins=n, color='#0D3592', rwidth=0.85)
    axs_gauss[0].axvline(np.mean(final_pos), color='k', linestyle='dashed', linewidth=0.8)
    axs_gauss[0].grid()
    axs_gauss[1].hist(final_vel, bins=n, color='#0D3592', rwidth=0.85)
    axs_gauss[1].axvline(np.mean(final_vel), color='k', linestyle='dashed', linewidth=0.8)
    axs_gauss[1].grid()
    if save:
        if os.path.isdir("./logs/" + folder_name) is False:
            temp_list = folder_name.split("/")
            fname = ''
            for i in range(len(temp_list) - 1):
                fname += temp_list[:i + 1][i]
                if os.path.isdir("./logs/" + fname) is False:
                    os.mkdir("./logs/" + fname)
                fname += "/"
        fig_gauss.savefig("./logs/" + folder_name + file_name + '.png', dpi=300, bbox_inches='tight')
        fig_gauss.savefig("./logs/" + folder_name + file_name + '.eps', format='eps')
    plt.draw()
    return [np.mean(final_pos), np.mean(final_vel), np.std(final_pos), np.std(final_vel)]


def plot_sigma_distribution(pos_, vel_, land_index, folder_name, file_name, lim_std3sigma, save=False):
    fig_dist, axs_dist = plt.subplots(1, 1)
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Altitude [m]')
    e1 = patches.Ellipse((0, 0), 2 * lim_std3sigma[1], 2 * lim_std3sigma[0],
                         angle=0, linewidth=2, fill=False, zorder=2,  edgecolor='red')
    e2 = patches.Ellipse((0, 0), 4 * lim_std3sigma[1] / 3, 4 * lim_std3sigma[0] / 3,
                         angle=0, linewidth=2, fill=False, zorder=2,  edgecolor='red')
    e3 = patches.Ellipse((0, 0), 2 * lim_std3sigma[1] / 3, 2 * lim_std3sigma[0] / 3,
                         angle=0, linewidth=2, fill=False, zorder=2,  edgecolor='red')

    axs_dist.add_patch(e1)
    axs_dist.add_patch(e2)
    axs_dist.add_patch(e3)
    plt.text(- lim_std3sigma[1] + 0.1, 0, r'$3 \sigma$', fontsize=15)
    plt.text(- 2 * lim_std3sigma[1] / 3 + 0.1, 0, r'$2 \sigma$', fontsize=15)
    plt.text(- lim_std3sigma[1] / 3 + 0.1, 0, r'$1 \sigma$', fontsize=15)
    for k in range(len(pos_)):
        plt.plot(vel_[k][land_index[k]], pos_[k][land_index[k]], 'bo')
    plt.grid()
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
        fig_dist.savefig("./logs/" + folder_name + file_name + '.eps', format='eps')
    plt.draw()
    return


def plot_state_vector(best_pos, best_vel, ini_best_individuals, end_best_individuals, save=False,
                      folder_name=None, file_name=None):
    fig_state = plt.figure()

    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Position [m]')
    for k in range(len(best_pos)):
        plt.plot(np.array(best_vel[k]), best_pos[k], lw=0.8)
        plt.scatter(np.array(best_vel[k])[ini_best_individuals[k]], np.array(best_pos[k])[ini_best_individuals[k]],
                    s=15, facecolors='none', edgecolors='g', label='StartBurnTime')
        plt.scatter(np.array(best_vel[k])[end_best_individuals[k]],
                    np.array(best_pos[k])[end_best_individuals[k]],
                    s=15, facecolors='none', edgecolors='r', label='EndBurnTime')
    plt.grid(True)
    if save:
        if save:
            if os.path.isdir("./logs/" + folder_name) is False:
                temp_list = folder_name.split("/")
                fname = ''
                for i in range(len(temp_list) - 1):
                    fname += temp_list[:i + 1][i]
                    if os.path.isdir("./logs/" + fname) is False:
                        os.mkdir("./logs/" + fname)
                    fname += "/"
        fig_state.savefig("./logs/" + folder_name + file_name + '.png', dpi=300, bbox_inches='tight')
        fig_state.savefig("./logs/" + folder_name + file_name + '.eps', format='eps')
    plt.draw()


def plot_performance(performance_list, n_thrusters, save=True, folder_name=None, file_name=None):
    fig_perf, axs_perf = plt.subplots(2, 1)
    axs_perf[0].set_ylabel('Landing position [m]')
    axs_perf[0].set_xlabel('Number of thrusters')
    axs_perf[0].grid()
    axs_perf[0].errorbar(np.arange(1, 1 + n_thrusters), np.array(performance_list)[:, 0],
                             yerr=np.array(performance_list)[:, 2], fmt='-o', capsize=5, color='g', ecolor='g')

    axs_perf[1].set_ylabel('Landing velocity [m/s]')
    axs_perf[1].set_xlabel('Number of thrusters')
    axs_perf[1].grid()
    axs_perf[1].errorbar(np.arange(1, 1 + n_thrusters), np.array(performance_list)[:, 1],
                             yerr=np.array(performance_list)[:, 3], fmt='-o', capsize=5, ecolor='g', color='g')

    if save:
        if os.path.isdir("./logs/" + folder_name) is False:
            temp_list = folder_name.split("/")
            fname = ''
            for i in range(len(temp_list) - 1):
                fname += temp_list[:i + 1][i]
                if os.path.isdir("./logs/" + fname) is False:
                    os.mkdir("./logs/" + fname)
                fname += "/"
    fig_perf.savefig("./logs/" + folder_name + file_name + '.png', dpi=300, bbox_inches='tight')
    fig_perf.savefig("./logs/" + folder_name + file_name + '.eps', format='eps')
    plt.show()
    return


def close_plot():
    plt.close('all')


def plot_thrust(time, thrust):
    plt.figure()
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [N]')
    plt.plot(time, thrust)
    plt.grid()
    plt.show()
    return


def plot_polynomial_function(degree):
    y0 = 2000
    v0 = 0
    g_center_body = -1.62  # moon
    a0 = g_center_body / 2
    poly = [a0, v0, y0]
    root = np.roots(poly)
    tf = max(root)
    dt = 0.1
    t = np.linspace(0, tf, 100)
    y_free_trajectory = y0 + v0 * t + a0 * t ** 2
    v_free_velocity = v0 + g_center_body * t

    v_aux = np.linspace(0, min(v_free_velocity), 100)
    y_aux = []
    y_sum = 0
    for i in range(1, degree + 1):
        y_ = -(-1) ** (i - 1) * v_aux ** i
        y_sum += y_
        y_aux.append(y_)

    plt.figure()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Position [m]')
    plt.plot(v_free_velocity, y_free_trajectory, label='Free fall', lw=0.8)
    plt.ylim([-100, 1.5 * y0])
    for i in range(degree):
        text = 'p = ' + str(i + 1)
        plt.plot(v_aux, y_aux[i], label=text, lw=0.8)
    if degree > 1:
        plt.plot(v_aux, y_sum, label='Total', lw=0.8)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_polynomial_function(3)
