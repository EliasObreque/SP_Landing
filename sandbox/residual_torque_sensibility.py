"""
Created by Elias Obreque
Date: 05-12-2023
email: els.obrq@gmail.com
"""
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn

from tools.pso import PSOStandard
# rmse
from sklearn.metrics import mean_squared_error
from core.module.Module import Module
from core.thrust.thruster import Thruster
from core.thrust.thrustProperties import second_thruster
from core.thrust.propellant.propellantProperties import second_propellant
from tools.Viewer import plot_state_solution, plot_normal_tangent_velocity, plot_pso_result


mass_0 = 24.0
rm = 1.738e6
inertia_0 = 1 / 12 * mass_0 * (0.2 ** 2 + 0.3 ** 2)
dt = 0.01
n_thruster = 2
t_end = 60*0.6
dataset = []
state_ = [np.array([0, rm + 5000]),
          np.array([0, -50]),
          0,
          0]
folder = "./logs/rw_tuning/"
name_ = "rw_pid"

thruster_pos = np.array([[-0.11, -0.0887], [0.11, -0.0887]])
thruster_ang = np.random.normal(0, np.deg2rad(0.5), size=(len(thruster_pos)))

thruster_properties_ = [copy.deepcopy(second_thruster), copy.deepcopy(second_thruster)]
propellant_properties_ = [copy.deepcopy(second_propellant), copy.deepcopy(second_propellant)]

list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [deg]", "Angular velocity [deg/s]",
             "Inertia [kgm2]", "Thrust [N]", "Torque [mNm]", "Energy [kJ]", "beta [-]", "Angle Error [deg]",
             "Angular velocity Error [deg/s]", "RW velocity [deg/s]",
             "RW Torque [mNm]"]


def sim_model(sig_ang=0, sig_d=0, sig_isp=0, bias_isp=0, dead_time=0, pid_gain: list = (1.0, 1.0, 1.0)):
    # Thruster
    second_thruster['max_ignition_dead_time'] = dead_time
    # Propellant
    second_propellant['isp_noise_std'] = sig_isp
    second_propellant['isp_bias_std'] = bias_isp

    thruster_pos_ = np.array([[-0.11, -0.0887], [0.11, -0.0887]])
    thruster_ang_ = np.random.normal(0, np.deg2rad(sig_ang), size=(len(thruster_pos)))

    thruster_ang_ += np.array([-0 * np.pi / 180, 0 * np.pi / 180])
    thruster_pos_ += np.random.normal(0, sig_d, size=2)

    module = Module(mass_0, inertia_0, state_, 0, 0,
                    thruster_pos_, thruster_ang_, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)
    module.set_control_function([4990, 4990])
    module.control_pid.set_gain(pid_gain[0], pid_gain[1], pid_gain[2])
    historical_state_ = module.simulate(t_end, low_step=dt, progress=False, only_thrust=False, force_step=True,
                                        force_mode=1)
    historical_state_[3] = np.rad2deg(historical_state_[3])
    historical_state_[4] = np.rad2deg(historical_state_[4])
    historical_state_[7] = np.array(historical_state_[7]) * 1000
    historical_theta_error_ = np.rad2deg(module.historical_theta_error)
    historical_omega_error_ = np.rad2deg(module.historical_omega_error)
    historical_rw_velocity_ = np.rad2deg(module.rw_model.historical_rw_velocity)
    historical_state_.insert(-1, historical_theta_error_)
    historical_state_.insert(-1, historical_omega_error_)
    historical_state_.insert(-1, historical_rw_velocity_)
    torque_rw_ = np.array(module.rw_model.historical_rw_torque)
    ctrl_torque_ = np.array(module.rw_model.historical_rw_ctrl_torque)
    historical_state_.insert(-1, np.vstack([torque_rw_, ctrl_torque_]).T * 1e3)
    return historical_state_


def cost_model(gain_):
    gain_ = gain_[0]
    historical_state_ = sim_model(0.5, 0.0001, 10.83, 3.25, 0.5, pid_gain=gain_)
    # angular error in deg
    mech_cost = historical_state_[-2]
    error = historical_state_[10]
    cost = mean_squared_error(np.zeros_like(error), error, squared=False) + np.max(np.abs(error))#+ np.mean(np.asarray(mech_cost) ** 2)
    return cost, historical_state_


if __name__ == '__main__':
    hist_list = []
    # noise evaluation iterative args = (sig_ang=0, sig_d=0, sig_isp=0, bias_isp=0, dead_time=0)
    args_list = np.zeros((6, 5))
    args_list[0] = [0.5, 0, 0, 0, 0]
    args_list[1] = [0, 0.0001, 0, 0, 0]
    args_list[2] = [0, 0, 10.83, 0, 0]
    args_list[3] = [0, 0, 0, 3.25, 0]
    args_list[4] = [0, 0, 0, 0, 0.5]
    args_list[5] = [0.5, 0.0001, 10.83, 3.25, 0.5]

    train_ = True
    if train_:
        pso_rw = PSOStandard(cost_model, n_particles=30, n_steps=5)
        pso_rw.initialize([[1e-6, 600],
                           [0, 10],
                           [1e-6, 300]])
        final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_rw.optimize(clip=True, tol=1e-10)

        plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name_ + "_train", plot_flag=True)
        plot_state_solution([best_state], list_name, folder, name_ + "_train", plot_flag=True)
        plot_normal_tangent_velocity([best_state], folder, name_ + "_train", plot_flag=True)
        plt.show()

    statistic_torque = {"Angle": [], "Displacement": [], "Thrust Bias": [], "Thrust Noise": [], "Dead time": [], "All": []}
    for ki, elem in enumerate([args_list[-1]]):
        print(elem)
        # plt.figure()
        # plt.xlabel("Time [s]", fontsize=14)
        # plt.ylabel("Residual Torque [mNm]", fontsize=14)
        # plt.grid()

        for i in range(10):
            print(i)
            if train_:
                gain = pso_rw.gbest_position
                historical_state = sim_model(pid_gain=gain)
            else:
                historical_state = sim_model(pid_gain=[10, 0.0, 15])

            hist_list.append(historical_state)

            # plt.plot(historical_state[-1], historical_state[7], lw=0.7)

            statistic_torque[list(statistic_torque.keys())[ki]] += list(historical_state[7])
        print(np.mean(statistic_torque[list(statistic_torque.keys())[ki]]),
              np.std(statistic_torque[list(statistic_torque.keys())[ki]]))
    seaborn.set(style='whitegrid')
    plt.figure()
    data = {"Residual Torque [mNm]":  statistic_torque,
            "Uncertainties": list(statistic_torque.keys())}

    ax1 = seaborn.violinplot(statistic_torque, inner_kws=dict(box_width=5, whis_width=3, color=".8"))
    plt.xlabel('Uncertainties', fontsize=14)
    plt.ylabel('Residual Torque [mNm]', fontsize=14)
    ax2 = seaborn.displot(statistic_torque, kind="kde", multiple="stack", fill=True)
    plt.xlabel('Residual Torque [mNm]', fontsize=14)
    plt.ylabel('Density - Residual Torque', fontsize=14)
    plot_state_solution(hist_list, list_name, folder, name_, plot_flag=True)
    plot_normal_tangent_velocity(hist_list, folder, name_, plot_flag=True)
    plt.show()
