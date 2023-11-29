"""
Created by Elias Obreque
Date: 23-11-2023
email: els.obrq@gmail.com
"""
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import pickle

from core.module.Module import Module
from core.thrust.thrustProperties import second_thruster
from core.thrust.propellant.propellantProperties import second_propellant
from tools.mathtools import propagate_rv_by_ang
from tools.pso import PSOStandard
from tools.Viewer import plot_orbit_solution, plot_state_solution, plot_pso_result, plot_general_solution

import json

np.random.seed(42)

mass_0 = 24.0
inertia_0 = 1 / 12 * mass_0 * (0.2 ** 2 + 0.3 ** 2)

mu = 4.9048695e12  # m3s-2
rm = 1.738e6
ra = 2e6
rp = 2e6
a = 0.5 * (ra + rp)
ecc = 1 - rp / a
b = a * np.sqrt(1 - ecc ** 2)
vp = np.sqrt(2 * mu / rp - mu / a)
va = np.sqrt(2 * mu / ra - mu / a)
p_ = rp * (1 + ecc)
f_0 = -np.pi / 2
r_0 = p_ / (1 + ecc * np.cos(f_0))

rot_90 = np.array([[0, -1], [1, 0]])
position = rot_90 @ np.array([np.cos(f_0), np.sin(f_0)]) * r_0
velocity = np.sqrt(mu / p_) * rot_90 @ np.array([-np.sin(f_0), (ecc + np.cos(f_0))])

theta = np.array(270.0 * np.deg2rad(1))
omega = np.array(0.0)
dt = 0.1
tf = min(15000000, 2 * np.pi / np.sqrt(mu) * a ** (3 / 2))
state = [position, velocity, theta, omega]

# TARGET
h_target = rm + 2e3
rp_target = 2e6

energy_target = -mu / h_target

thruster_pos = np.array([[-0.06975, -0.0887], [-0.06975, 0.0887]])
thruster_pos += np.random.normal(0, 0.0001, size=np.shape(thruster_pos))
thruster_ang = np.random.normal(0, np.deg2rad(0.5), size=(len(thruster_pos)))

thruster_properties_ = [copy.deepcopy(second_thruster), copy.deepcopy(second_thruster)]
propellant_properties_ = [copy.deepcopy(second_propellant), copy.deepcopy(second_propellant)]

# noise_state = [np.random.normal(0, 0, size=2), np.random.normal(0, 0, size=2)]
# noise_isp = [np.random.normal(0, propellant_properties_[0]['isp_bias_std'])]
# dead_time = [np.random.normal(0, thruster_properties_[0]['max_ignition_dead_time'])]

list_name = ["Position [m]", "Velocity [km/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
             "Inertia [kgm2]", "Thrust [N]", "Torque [Nm]", "Energy [J]"]
list_gain = [-1]
folder = "logs/neutral/train/"
name_ = "mass_opt_vf_"


def descent_optimization(modules_setting_):
    gain_file = open(folder + name_ + "gain.txt", "r")
    line_read = gain_file.readlines()
    gain_normal_vel = float(float(line_read[-1]))
    gain_file.close()

    # read for pkl file
    with open(folder + name_ + "uncertainties.pkl", "rb") as read_file:
        uncertainties = dict(pickle.load(read_file))
        read_file.close()

    control_set_ = [modules_setting_[0], modules_setting_[0]]

    pos_temp = state[0] + uncertainties["state"][0]


    state_ = [state[0] + uncertainties["state"][0],
              state[1] + uncertainties["state"][1],
              state[2],
              state[3]]

    mass_, inertia_ = mass_0, inertia_0

    state_[0], state_[1] = propagate_rv_by_ang(state_[0], state_[1], modules_setting_[0], mu)
    module = Module(mass_, inertia_, state_,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)

    module.set_thrust_design([modules_setting_[1], modules_setting_[1]],
                             [modules_setting_[2], modules_setting_[2]])

    module.set_thrust_bias(list(uncertainties["isp"]), dead_time=list(uncertainties["dead"]))

    module.set_control_function(control_set_)
    historical_state = module.simulate(tf, low_step=0.1, progress=False, only_thrust=True)

    # COST
    r_state = np.array([np.linalg.norm(elem) for elem in historical_state[0]])
    v_state = np.array([np.linalg.norm(elem) for elem in historical_state[1]])
    mass_state = np.array([np.linalg.norm(elem) for elem in historical_state[2]])
    state_energy = historical_state[8]
    # error = (state_energy[-1] - energy_target) ** 2 / mass_state[-1]

    ang = np.arctan2(historical_state[0][-1][1], historical_state[0][-1][0])
    v_t_n = np.array([[np.cos(ang - np.pi / 2), -np.sin(ang - np.pi / 2)],
                      [np.sin(ang - np.pi / 2), np.cos(ang - np.pi / 2)]]).T @ historical_state[1][-1]

    if gain_normal_vel >= 0:
        error = state_energy[-1] + gain_normal_vel * v_t_n[0] ** 2
    else:
        error = v_t_n[0] ** 2
    error /= mass_state[-1]
    # error += 100000 if module.dynamics.isTouchdown() else 1
    return error, historical_state


if __name__ == '__main__':
    n_step = 100
    n_par = 30
    stage = "D"
    plot_flag = True

    range_variables = [(1.2 * np.pi, 1.8 * np.pi),
                       (0.05, 0.065),
                       (0.1, 0.2)
                       ]

    for i, gain in enumerate(list_gain):
        gain_file = open(folder + name_ + "gain.txt", "w")
        gain_file.write("{}\n".format(gain))
        gain_file.close()
        gain_normal_vel = gain
        hist_list = []
        for j in range(10):
            print(i, j)
            noise_state_ = [np.random.normal(0, (100, 100)), np.random.normal(0, (5, 5))]
            noise_isp_ = [np.random.normal(0, prop['isp_bias_std']) for prop in propellant_properties_]
            dead_time_ = [np.random.uniform(0, thr['max_ignition_dead_time']) for thr in thruster_properties_]

            # save in a pkl file
            # with open(folder + name_ + "uncertainties.pkl", "wb") as outfile:
            #     pickle.dump({"state": noise_state_,
            #                  "isp": noise_isp_,
            #                  "dead": dead_time_}, outfile)
            #     outfile.close()

            name = name_ + str(gain) if gain >= 0 else name_ + "full"
            name += "_{}".format(j)
            # pso_algorithm = PSOStandard(descent_optimization, n_particles=n_par, n_steps=n_step)
            # pso_algorithm.initialize(range_variables)
            #
            # init_time = time.time()
            # final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_algorithm.optimize(clip=True)
            # end_time = time.time()
            # print("Optimization Time: {}".format((end_time - init_time) / 60))
            # modules_setting = pso_algorithm.gbest_position
            modules_setting = [2 * np.pi, 0.065, 0.2]

            state = [position, velocity, theta, omega]
            state_ = [state[0] + noise_state_[0],
                      state[1] + noise_state_[1],
                      state[2],
                      state[3]]

            module = Module(mass_0, inertia_0, state_,
                            thruster_pos, thruster_ang, thruster_properties_,
                            propellant_properties_, "2D", dt, training=True)
            control_set_ = [modules_setting[0], modules_setting[0]]
            module.set_thrust_bias(noise_isp_, dead_time=dead_time_)
            module.set_control_function(control_set_)
            module.set_thrust_design([modules_setting[1], modules_setting[1]],
                                     [modules_setting[2], modules_setting[2]])
            historical_state = module.simulate(15000, low_step=0.1, progress=False)
            # data = {'state': historical_state,
            #         'state_name': list_name,
            #         'best_cost': pso_algorithm.evol_best_fitness,
            #         'p_cost': pso_algorithm.evol_p_fitness,
            #         'best_part': pso_algorithm.historical_g_position,
            #         'hist_part': pso_algorithm.historical_position}

            # with open(folder + name + ".pkl", "wb") as data_handle:
            #     pickle.dump(data, data_handle)
            #     data_handle.close()

            historical_state[1] = np.array(historical_state[1]) / 1000
            hist_list.append(historical_state)

            # if plot_flag:
            #     # plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name, plot_flag=plot_flag)
            #     plot_state_solution(historical_state, list_name, folder, name, aux={8: energy_target},
            #                         plot_flag=plot_flag)

        if plot_flag:
            plot_orbit_solution(hist_list, ["orbit"], a, b, rp, folder, name_,
                                plot_flag=plot_flag)
            plot_general_solution(hist_list, ["general"], a, b, rp, folder, name_, plot_flag=plot_flag)
            plt.show()