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
from tools.Viewer import plot_orbit_solution, plot_state_solution, plot_pso_result, plot_general_solution, plot_normal_tangent_velocity

import json

np.random.seed(42)

mass_0 = 24.0
inertia_0 = 1 / 12 * mass_0 * (0.2 ** 2 + 0.3 ** 2)

mu = 4.9048695e12  # m3s-2
rm = 1.738e6
ra = 68e6
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

energy_target = -mu / rm

thruster_pos = np.array([[-0.06975, -0.0887], [0.06975, -0.0887]])
thruster_pos += np.random.normal(0, 0.0000, size=np.shape(thruster_pos))
thruster_ang = np.random.normal(0, np.deg2rad(0.0), size=(len(thruster_pos)))

thruster_properties_ = [copy.deepcopy(second_thruster), copy.deepcopy(second_thruster)]
propellant_properties_ = [copy.deepcopy(second_propellant), copy.deepcopy(second_propellant)]

list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
                 "Inertia [kgm2]", "Thrust [N]", "Torque [mNm]", "Energy [kJ]", "Angle Error [rad]",
                 "Angular velocity Error [rad/s]", "RW velocity [rad/s]",
                 "RW Torque [mNm]", "beta [-]"]
list_gain = [-1]
folder = "logs/neutral/attitude/"
name_ = "mass_opt_2_vf_"
sigma_r = 100
sigma_v = 10


def descent_optimization(modules_setting_):
    gain_file = open(folder + name_ + "gain.txt", "r")
    line_read = gain_file.readlines()
    gain_normal_vel = float(float(line_read[-1]))
    gain_file.close()

    control_set_ = [modules_setting_[0], modules_setting_[0]]

    r_, v_ = propagate_rv_by_ang(state[0], state[1], modules_setting_[0] - np.deg2rad(0.01), mu)
    state_ = [r_,
              v_,
              state[2],
              state[3]]

    mass_, inertia_ = mass_0, inertia_0
    module = Module(mass_, inertia_, state_, sigma_r, sigma_v,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)

    module.set_thrust_design([modules_setting_[1], modules_setting_[1]],
                             [modules_setting_[2], modules_setting_[2]])

    # module.set_thrust_bias(list(uncertainties["isp"]), dead_time=list(uncertainties["dead"]))

    module.set_control_function(control_set_)
    historical_state = module.simulate(tf, low_step=0.1, progress=False, only_thrust=True, force_step=True)

    # COST
    mass_state = np.array([np.linalg.norm(elem) for elem in historical_state[2]])
    state_energy = historical_state[8]

    ang = np.arctan2(historical_state[0][-1][1], historical_state[0][-1][0])
    v_t_n = np.array([[np.cos(ang - np.pi / 2), -np.sin(ang - np.pi / 2)],
                      [np.sin(ang - np.pi / 2), np.cos(ang - np.pi / 2)]]).T @ historical_state[1][-1]

    if gain_normal_vel >= 0:
        error = state_energy[-1] + gain_normal_vel * v_t_n[0] ** 2
    else:
        error = v_t_n[0] ** 2
    error /= mass_state[-1] ** 2
    return error, historical_state


if __name__ == '__main__':
    n_step = 5
    n_par = 5
    stage = "D"
    plot_flag = True

    range_variables = [(4, 6),
                       (0.03, 0.04),
                       (0.15, 0.2)
                       ]
    dataset = {}
    for i, gain in enumerate(list_gain):
        gain_file = open(folder + name_ + "gain.txt", "w")
        gain_file.write("{}\n".format(gain))
        gain_file.close()
        gain_normal_vel = gain
        hist_list = []
        name = name_ + str(gain) if gain >= 0 else name_ + "full"
        dataset[name] = []
        for j in range(1):
            print(i, j)
            name_int = name + "_{}".format(j)
            pso_algorithm = PSOStandard(descent_optimization, n_particles=n_par, n_steps=n_step)
            pso_algorithm.initialize(range_variables)

            init_time = time.time()
            final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_algorithm.optimize(clip=True)
            end_time = time.time()
            print("Optimization Time: {}".format((end_time - init_time) / 60))
            modules_setting = pso_algorithm.gbest_position

            state = [position, velocity, theta, omega]
            r_, v_ = propagate_rv_by_ang(state[0], state[1], modules_setting[0] - np.deg2rad(0.02), mu)
            state_ = [r_,
                      v_,
                      state[2],
                      state[3]]

            module = Module(mass_0, inertia_0, state_, sigma_r, sigma_v,
                            thruster_pos, thruster_ang, thruster_properties_,
                            propellant_properties_, "2D", dt, training=True)
            control_set_ = [modules_setting[0], modules_setting[0]]
            # module.set_thrust_bias(noise_isp_, dead_time=dead_time_)
            module.set_control_function(control_set_)
            module.set_thrust_design([modules_setting[1], modules_setting[1]],
                                     [modules_setting[2], modules_setting[2]])
            historical_state = module.simulate(800000, low_step=0.1, progress=False)

            historical_state.insert(-1, module.historical_theta_error)
            historical_state.insert(-1, module.historical_omega_error)
            historical_state.insert(-1, module.rw_model.historical_rw_velocity)
            historical_state.insert(-1, np.array(module.rw_model.historical_rw_torque) * 1e3)
            historical_state.insert(-1, module.get_betas_control())

            data = {'state': historical_state,
                    'state_name': list_name,
                    'best_cost': pso_algorithm.evol_best_fitness,
                    'p_cost': pso_algorithm.evol_p_fitness,
                    'best_part': pso_algorithm.historical_g_position,
                    'hist_part': pso_algorithm.historical_position}
            dataset[name].append(data)
            historical_state[1] = np.array(historical_state[1]) / 1000
            historical_state[7] = np.array(historical_state[7]) * 1000
            historical_state[8] = np.array(historical_state[8]) * 1000

            hist_list.append(historical_state)
            # new Agg
            if not plot_flag:
                plt.switch_backend('Agg')
            # plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name_int, plot_flag=plot_flag)

        with open(folder + name + ".pkl", "wb") as data_handle:
            pickle.dump(dataset, data_handle)
            data_handle.close()
        plot_state_solution(hist_list, list_name, folder, name, aux={8: energy_target}, plot_flag=plot_flag)
        plot_normal_tangent_velocity(hist_list, folder, name, plot_flag=plot_flag)
        plot_orbit_solution(hist_list, ["Orbit"], a, b, rp, folder, name, plot_flag=plot_flag)
        plot_general_solution(hist_list, ["General"], a, b, rp, folder, name, plot_flag=plot_flag)
        plt.show()
