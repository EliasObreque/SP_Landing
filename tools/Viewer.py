"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
from matplotlib import pyplot as plt
import pickle
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import patches
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image
from matplotlib.transforms import Affine2D
# plt.rcParams["font.family"] = "Times New Roman"

ra = 68e6
rp = 2e6
a = 0.5 * (ra + rp)
ecc = 1 - rp / a
b = a * np.sqrt(1 - ecc ** 2)
rm = 1.738e6


def plot_best_cost(evol_p_fitness, evol_best_fitness, folder=None, name=None):
    fig = plt.figure()
    # plt.rcParams['font.size'] = 13
    plt.plot(np.arange(1, len(evol_best_fitness) + 1), evol_best_fitness, 'red', lw=1)
    plt.plot(np.arange(1, len(evol_best_fitness) + 1), evol_p_fitness.T, '-.', color='blue', lw=0.5)
    plt.grid()
    plt.ylabel("Evaluation cost")
    plt.xlabel("Iteration")
    plt.yscale('log')
    plt.gca().yaxis.set_label_coords(-0.12, 0.5)
    plt.subplots_adjust(left=0.16, right=0.95, top=0.9)  # Puedes ajustar este valor según tus necesidades
    if folder is not None and name is not None:
        fig.savefig(folder + name + "_hist_cost.pdf", format='pdf')
    return fig


def plot_historical_position(historical_position, historical_g_position, folder=None, name=None):
    dim_var = np.shape(historical_position[0])[1]
    fig, axes = plt.subplots(dim_var, 1, sharex=True, figsize=(10, 5))
    plt.xlabel("Iteration")
    plt.subplots_adjust(left=0.16, right=0.95, top=0.95)  # Puedes ajustar este valor según tus necesidades
    if dim_var > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    max_iteration = len(historical_g_position)
    for i, ax in enumerate(axes):
        ax.set_ylabel("P {}".format(i + 1))
        ax.yaxis.set_label_coords(-0.12, 0.5)
        ax.plot(np.arange(1, max_iteration + 1),
                np.array(historical_position).T[i].T, '-.', lw=0.5, color='b')
        ax.plot(np.arange(1, max_iteration + 1),
                np.array(historical_g_position).T[i], lw=1.0, color='r')
        ax.grid()
    if folder is not None and name is not None:
        fig.savefig(folder + name + "_hist_pos.pdf", format='pdf')
    return fig


def plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder="", name="", plot_flag=True):
    fig_h = plot_historical_position(hist_pos, hist_g_pos, folder, name)
    fig_b = plot_best_cost(eval_pos, eval_g_pos, folder, name)
    if not plot_flag:
        plt.close(fig_h)
        plt.close(fig_b)


def plot_state_solution(min_state_full, list_name, folder=None, name=None, aux: dict = None, plot_flag=True):
    for i, min_state in enumerate(min_state_full[:-1]):
        fig = plt.figure()
        plt.grid()
        # plt.rcParams['font.size'] = 13
        plt.gca().yaxis.set_label_coords(-0.12, 0.5)
        plt.subplots_adjust(left=0.16, right=0.95, top=0.9)  # Puedes ajustar este valor según tus necesidades
        if list_name is not None:
            plt.ylabel(list_name[i])
        plt.xlabel("Time [min]")
        wind_time = np.array(min_state_full[-1]) / 60
        plt.plot(wind_time, min_state)
        # if "Thrust" in list_name[i]:
        #     thr_wind = np.argwhere(np.array(min_state) > 0)
        #     init = np.max([thr_wind.min() - 1, 0])
        #     stop = np.min([thr_wind.max() + 2, len(min_state)])
        #     axes = fig.add_axes([0.63, 0.58, 0.3, 0.3]) # left, bottom, width, height - en porcentajes
        #     axes.plot(wind_time[init:stop], min_state[init:stop])
        #     axes.set_ylim(0, 1.2 * max(min_state))
        #     plt.xticks(fontsize=8)
        #     plt.yticks(fontsize=8)
        #     axes.grid()
        if aux is not None:
            if i in list(aux.keys()):
                plt.hlines(aux[i], xmin=min(wind_time), xmax=max(wind_time), colors='red')

        if folder is not None and name is not None:
            fig.savefig(folder + name + "_" + list_name[i].split(" ")[0] + '.pdf', format='pdf')
        if not plot_flag:
            plt.close(fig)


def plot_orbit_solution(min_state_full, list_name, a_, b_, rp_, folder=None, name=None, h_target=None, plot_flag=True):
    fig_pso, ax_ = plt.subplots(1, 2, figsize=(12, 7))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax_pos = ax_[0]
    axes = ax_[1]
    ellipse = Ellipse(xy=(0, -(a_ - rp_) * 1e-3), width=b_ * 2 * 1e-3,
                      height=2 * a_ * 1e-3,
                      edgecolor='r', fc='None', lw=0.7)
    ellipse_moon = Ellipse(xy=(0, 0), width=2 * rm * 1e-3, height=2 * rm * 1e-3, fill=True,
                           edgecolor='black', fc='None', lw=0.4)
    if h_target is not None:
        ellipse_target = Ellipse(xy=(0, 0), width=2 * (h_target * 1e-3),
                                 height=2 * (h_target * 1e-3),
                                 edgecolor='green', fc='None', lw=0.7)
        ax_pos.add_patch(ellipse_target)

    data = Image.open("moon-58-1024x1024.png")
    data = np.asarray(data)[61:-75, 68:-68]
    data_ = Image.fromarray(data)
    data_ = data_.resize((int(2 * rm * 1e-3), int(2 * rm * 1e-3)))
    translacion_x = -int(rm * 1e-3)  # Ajusta este valor según sea necesario
    translacion_y = -int(rm * 1e-3)  # Ajusta este valor según sea necesario
    transformacion = Affine2D().translate(translacion_x, translacion_y)
    ax_pos.imshow(data_, transform=transformacion + ax_pos.transData)
    # ax_pso.add_patch(ellipse)
    ax_pos.add_patch(ellipse_moon)
    ax_pos.set_ylabel("Y Position [km]", fontsize=14)
    ax_pos.set_xlabel("X Position [km]", fontsize=14)
    ax_pos.set_xlim((-rm * 1e-3 - 15000, rm * 1e-3 + 15000))
    ax_pos.set_ylim((-rm * 1e-3 - 70000, rm * 1e-3 + 1000))
    ax_pos.grid(linestyle='--', linewidth=0.5)

    for min_state in min_state_full:
        wind_time = np.array(min_state[-1]) / 60
        x_pos = [elem[0] * 1e-3 for elem in min_state[0]]
        y_pos = [elem[1] * 1e-3 for elem in min_state[0]]
        ax_pos.plot(x_pos, y_pos, 'b', lw=0.7)
        axes.plot(x_pos, y_pos, 'b', lw=1)

    ellipse_moon = Ellipse(xy=(0, 0), width=2 * rm * 1e-3, height=2 * rm * 1e-3, fill=True,
                           edgecolor='black', fc='None', lw=0.4)
    axes.add_patch(ellipse_moon)
    axes.imshow(data_, transform=transformacion + axes.transData, aspect='auto')
    axes.set_xlim(-2500, 2500)
    axes.set_ylim(-2500, 2500)
    axes.grid()
    axes.set_ylabel("Y Position [km]", fontsize=14)
    axes.set_xlabel("X Position [km]", fontsize=14)
    ax_pos.indicate_inset_zoom(axes)
    ax_pos.set_position([0.1, 0.1, 0.6, 0.8])
    plt.tight_layout()
    if folder is not None and name is not None:
        fig_pso.savefig(folder + name + "_" + list_name[0].split(" ")[0] + '.pdf', format='pdf')
    if not plot_flag:
        plt.close(fig_pso)


def plot_general_solution(min_state_full, list_name, a_, b_, rp_, folder=None, name=None, h_target=None, plot_flag=True):
    fig_pso, ax_pso = plt.subplots(2, 2)
    # plt.rcParams['font.size'] = 13
    ax_pso = ax_pso.flatten()
    ax_pso[0].set_ylabel("Radial Velocity [km/s]")
    ax_pso[0].set_xlabel("Time [min]")
    ax_pso[0].grid()
    ax_pso[1].set_ylabel("Normal Velocity [km/s]")
    ax_pso[1].set_xlabel("Time [min]")
    ax_pso[1].grid()
    ax_pso[2].set_ylabel("Altitude [km]")
    ax_pso[2].set_xlabel("Time [min]")
    ax_pso[2].grid()
    ax_pso[3].set_ylabel("Normalized velocity [km/s]")
    ax_pso[3].set_xlabel("Time [min]")
    ax_pso[3].grid()

    for min_state in min_state_full:
        wind_time = np.array(min_state[-1]) / 60
        x_pos = [elem[0] * 1e-3 for elem in min_state[0]]
        y_pos = [elem[1] * 1e-3 for elem in min_state[0]]

        ang_rot = [np.arctan2(y_, x_) for x_, y_ in zip(x_pos, y_pos)]
        v_t_n = [np.array([[np.cos(ang - np.pi/2), -np.sin(ang - np.pi/2)],
                           [np.sin(ang - np.pi/2), np.cos(ang - np.pi/2)]]).T @ v_
                 for ang, v_ in zip(ang_rot, min_state[1])]

        ax_pso[0].plot(wind_time, np.array(v_t_n)[:, 1], 'b', lw=0.7)
        ax_pso[1].plot(wind_time, np.array(v_t_n)[:, 0], 'b', lw=0.7)
        ax_pso[2].plot(wind_time, np.sqrt(np.array(x_pos)**2 + np.array(y_pos)**2) - rm * 1e-3, 'b', lw=0.7)
        ax_pso[3].plot(wind_time, [np.linalg.norm(v_) for v_ in min_state[1]], 'b', lw=0.7)
    plt.tight_layout()
    if folder is not None and name is not None:
        fig_pso.savefig(folder + name + "_" + list_name[0].split(" ")[0] + '.pdf', format='pdf')
    if not plot_flag:
        plt.close(fig_pso)


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
    plt.ylabel('thrust (N)')
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
    plt.show()


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
    axs_best[1, 1].set_ylabel('thrust [N]')
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
        if os.path.isdir("logs/" + folder_name) is False:
            temp_list = folder_name.split("/")
            fname = ''
            for i in range(len(temp_list) - 1):
                fname += temp_list[:i + 1][i]
                if os.path.isdir("./logs/" + fname) is False:
                    os.mkdir("./logs/" + fname)
                fname += "/"
        fig_best.savefig("./logs/" + folder_name + file_name + '.png', dpi=300, bbox_inches='tight')
        fig_best.savefig("./logs/" + folder_name + file_name + '.eps', format='eps', bbox_inches='tight')
    return


def plot_distribution(pos_, vel_, land_index, folder_name=None, file_name=None, save=False):
    fig_dist, axs_dist = plt.subplots(1, 2)
    axs_dist[0].set_xlabel('Altitude [m]')
    axs_dist[1].set_xlabel('Velocity [m/s]')
    final_pos = [pos_[k][land_index[k]] for k in range(len(pos_))]
    final_vel = [vel_[k][land_index[k]] for k in range(len(pos_))]
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
        fig_dist.savefig("./logs/" + folder_name + file_name + '.eps', format='eps', bbox_inches='tight')
    return


def plot_state_vector(best_pos, best_vel, ini_best_individuals, end_best_individuals, save=False,
                      folder_name=None, file_name=None):
    fig_state = plt.figure()

    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Position [m]')
    for k in range(len(best_pos)):
        plt.plot(np.array(best_vel[k]), best_pos[k], lw=0.6)
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
        fig_state.savefig("./logs/" + folder_name + file_name + '.eps', format='eps', bbox_inches='tight')


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
        fig_perf.savefig("./logs/" + folder_name + file_name + '.eps', format='eps', bbox_inches='tight')
    return


def plot_dv_req():
    import matplotlib.pyplot as plt

    mu = 4.9048695e12  # m3s-2
    rm = 1.738e6
    ra = 68e6
    rp = 2e6
    a = 0.5 * (ra + rp)

    v0 = np.sqrt(2 * mu / ra - mu / a)
    a1 = 0.5 * ra
    v1 = np.sqrt(2 * mu / ra - mu / a1)
    dv1 = v0 - v1
    v2 = np.sqrt(mu / rm)
    dv2 = v2 - v1
    dv3 = v2
    dvt = dv1 + dv2 + dv3
    print(dvt)
    r = np.linspace(rp, ra, 200)
    ve = np.sqrt(mu * (2 / r - 1 / a))
    vf = np.sqrt(2 * mu * (1 / rm - 1 / r))
    v = ve + vf
    print(np.sqrt(mu * (2 / rm - 1 / a)))
    plt.figure()
    plt.ylabel(r"$\Delta v$")
    plt.plot(r * 1e-3, ve, label=r'$\Delta v_e$', lw=0.8)
    plt.plot(r * 1e-3, vf, label=r'$\Delta v_f$', lw=0.8)
    plt.plot(r * 1e-3, v, label=r'$\Delta v$', lw=0.8)
    plt.xlabel(r'Distance $R$ from moon mass center [km]')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    mf = 17.9
    Isp = np.arange(260, 320, 20)
    ge = 9.81
    mass = [24.0 * (1 - np.exp(-v / (Isp_i * ge))) for Isp_i in Isp]
    plt.figure()
    # plt.hlines(mf, rp * 1e-3, ra * 1e-3, color='r', linestyles='-', lw=0.8)
    # plt.hlines(mf * 0.7, rp * 1e-3, ra * 1e-3, color='b', linestyles='--', lw=1, label='Amateur (x0.7)')
    # plt.hlines(mf * 0.8, rp * 1e-3, ra * 1e-3, color='g', linestyles='--', lw=1, label='Space Shuttle SRB (x0.8)')
    # plt.hlines(mf * 0.9, rp * 1e-3, ra * 1e-3, color='m', linestyles='--', lw=1, label='SS - 520 JAXA (x0.9)')
    [plt.plot(r * 1e-3, mass[i], color='k', lw=0.8) for i in range(len(Isp))]
    [plt.text(r[0] * 1e-3, mass[i][0], r'$I_{sp}$=' + str(Isp[i]) + " s", rotation=-0, ha="left", va="center",
              bbox=dict(boxstyle="round",
                        ec=(0., 0.0, 0.0),
                        fc=(1., 1, 1),
                        )
              ) for i in range(len(Isp))]
    plt.grid()
    plt.legend()
    # int(len(r) * 0.5)
    plt.ylabel('Mass required: $m_p$ [kg]')
    plt.xlabel(r'Distance $R$ from moon mass center [km]')
    plt.tight_layout()
    plt.show()
    return


def close_plot():
    plt.close('all')


def plot_thrust(time, thrust, thrust_free=None, names=None, dead=0):
    plt.figure()
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [N]')
    # plt.ylim(0, 1.5)
    plt.plot(np.array(time) + dead, thrust)
    if thrust_free is not None:
        plt.plot(time, thrust_free)
    plt.grid()
    if names is not None:
        plt.legend(names)
    return


def plot_thrust_beta(time, thrust, beta, folder_name=None, file_name=None):
    fig_ = plt.figure()
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('thrust [N]')
    plt.plot(time[0], thrust[0], 'k', label='thrust')
    plt.legend()
    plt.twinx()
    plt.plot(time[0], beta[0], label=r'$\beta(t)$')
    plt.ylabel(r'$\beta(t)$')
    plt.axhline(0, color='r')
    plt.legend()
    fig_.savefig("./logs/" + folder_name + file_name + '.png', dpi=300, bbox_inches='tight')
    fig_.savefig("./logs/" + folder_name + file_name + '.eps', format='eps', bbox_inches='tight')
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


def compare_performance():
    import json
    folder_name = "../logs/Only_GA_all/regressive/"
    folder_name += "2022-02-08T02-30-34/"
    file_name = "eva_reg_performance_data.json"
    f1 = folder_name + file_name

    folder_name = "../logs/Only_GA_all/progressive/"
    folder_name += "2022-02-08T02-31-21/"
    file_name = "eva_pro_performance_data.json"
    f2 = folder_name + file_name

    folder_name = "../logs/Only_GA_all/neutral/"
    folder_name += "2022-02-08T02-30-59/"
    file_name = "eva_neu_performance_data.json"
    f3 = folder_name + file_name
    file_noise = [f1, f2, f3]
    performance = []
    for name_file in file_noise:
        f = open(name_file)
        data = json.load(f)
        performance.append([data['mean_pos'], data['mean_vel'], data['std_pos'], data['std_vel']])
    n_thrusters = len(performance[0][0])
    print("N engines: ", n_thrusters)
    fig_perf, axs_perf = plt.subplots(2, 1)
    axs_perf[0].set_ylabel('Landing position [m]')
    axs_perf[0].set_xlabel('Number of thrusters')
    axs_perf[0].grid()
    for performance_list in performance[:3]:
        axs_perf[0].errorbar(np.arange(1, 1 + n_thrusters), np.array(performance_list)[0],
                             yerr=np.array(performance_list)[2], fmt='-o', capsize=5) #, color='g', ecolor='g')
    for performance_list in performance[3:]:
        axs_perf[0].errorbar(np.arange(1, 1 + n_thrusters - 1), np.array(performance_list)[0],
                             yerr=np.array(performance_list)[2], fmt='--o', capsize=5) #, color='g', ecolor='g')

    axs_perf[1].set_ylabel('Landing velocity [m/s]')
    axs_perf[1].set_xlabel('Number of thrusters')
    axs_perf[1].grid()
    for performance_list in performance[:3]:
        axs_perf[1].errorbar(np.arange(1, 1 + n_thrusters), np.array(performance_list)[1],
                             yerr=np.array(performance_list)[3], fmt='-o', capsize=5)#, ecolor='g', color='g')
    for performance_list in performance[3:]:
        axs_perf[1].errorbar(np.arange(1, 1 + n_thrusters), np.array(performance_list)[1],
                             yerr=np.array(performance_list)[3], fmt='--o', capsize=5)#, ecolor='g', color='g')
    plt.legend()
    return


def isp_vacuum():
    gamma = 1.3
    ratio_p = 1 / np.linspace(10, 100)
    a = (2 / (gamma + 1)) ** ((gamma + 1)/(gamma - 1))
    gamma_upper = np.sqrt(a * gamma)
    b = 2 * gamma ** 2 / (gamma - 1)
    print(a, b)
    c = (1 - ratio_p ** ((gamma - 1) / gamma))
    cf_e = np.sqrt(b * a * c)

    ae_at = 1 / np.sqrt(b/gamma * ratio_p ** (2 / gamma) * c)
    #ratio = ae_at * ratio_p / 9.8
    ratio = 1 + 1/((ratio_p ** ((1 - gamma)/gamma)) - 1) * (gamma - 1)/(2 * gamma)

    plt.figure()
    plt.plot(1/ratio_p, ratio, 'k', lw=1)
    #plt.plot(ratio_p, ratio_2)
    plt.xlabel(r"$(P_c/P_e)$")
    plt.ylabel(r"$(I_{sp}^v/I_{sp}^e)$")
    plt.tight_layout()
    plt.grid()
    plt.show()
    return


if __name__ == '__main__':
    # plot_polynomial_function(3)
    # compare_performance()
    # plot_dv_req()
    # isp_vacuum()
    # data = open("../logs/plane/test", 'rb')
    h_target = rm + 2e3
    rp_target = 2e6
    mu = 4.9048695e12  # m3s-2
    energy_target = -mu / h_target
    folder = "../sandbox/logs/neutral/train/"
    list_name = ["Position [m]", "Velocity [km/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
                 "Inertia [kgm2]", "Thrust [N]", "Torque [Nm]", "Energy [J]"]
    # open all file *.pkl with import os
    plot_flag = True
    hist = []
    for file in os.listdir(folder):
        if "pkl" in file and 'full' in file:
            print(file)
            data = open(folder + file, 'rb')
            name = file.split('.')[0]
            data_loaded = dict(pickle.load(data))
            hist_pos, hist_g_pos = data_loaded['hist_part'], data_loaded['best_part']
            historical_state = data_loaded['state']
            eval_pos, eval_g_pos = data_loaded['p_cost'], data_loaded['best_cost']
            historical_state[1] = np.array(historical_state[1]) * 1e-3
            hist.append(historical_state)
            plot_pso_result(hist_pos, hist_g_pos, np.array(eval_pos), eval_g_pos, folder, name, plot_flag=plot_flag)
            plot_state_solution(historical_state, list_name, folder, name, aux={8: energy_target},
                                plot_flag=plot_flag)
    plot_orbit_solution(hist, ["orbit"], a, b, rp, folder, "all_vf",
                        h_target=h_target, plot_flag=plot_flag)

    plt.show()


