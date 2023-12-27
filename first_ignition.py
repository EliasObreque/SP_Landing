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
from core.thrust.thrustProperties import second_thruster, main_thruster, third_thruster
from core.thrust.propellant.propellantProperties import second_propellant, main_propellant, third_propellant
from tools.mathtools import propagate_rv_by_ang
from tools.pso import PSOStandard
from tools.Viewer import plot_orbit_solution, plot_state_solution, plot_pso_result, plot_general_solution, \
    plot_normal_tangent_velocity

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
ENTRY_MODE = 0
DESCENT_MODE = 1
LANDING_MODE = 2

list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
             "Inertia [kgm2]", "Thrust [N]", "Torque [mNm]", "Energy [kJ]", "beta [-]", "Angle Error [rad]",
             "Angular velocity Error [rad/s]", "RW velocity [rad/s]",
             "RW Torque [mNm]"]

folder = "logs/neutral/descent/"
name_ = "mass_opt_vf"
sigma_r = 100
sigma_v = 10


def entry_optimization(modules_setting_):
    args_ = modules_setting_[1]
    noise_ = modules_setting_[2]
    modules_setting_ = modules_setting_[0]

    thruster_pos = np.array([[-0.06975, -0.0887], [0.06975, -0.0887]])
    thruster_pos += noise_[0]
    thruster_ang = noise_[1] * np.deg2rad(1)
    thruster_ang += np.array([-25 * np.pi / 180, 25 * np.pi / 180])

    thruster_properties_ = [copy.deepcopy(second_thruster), copy.deepcopy(second_thruster)]
    propellant_properties_ = [copy.deepcopy(second_propellant), copy.deepcopy(second_propellant)]

    control_set_ = [modules_setting_[0], modules_setting_[0]]

    r_, v_ = propagate_rv_by_ang(state[0], state[1], modules_setting_[0] - np.deg2rad(0.00), mu)
    state_ = [r_ + noise_[2],
              v_ + noise_[3],
              state[2],
              state[3]]
    mass_, inertia_ = mass_0, inertia_0
    module = Module(mass_, inertia_, state_, 0, 0,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)

    # module.set_thrust_design([0.065, 0.065],
    #                          [0.2, 0.2])
    module.set_thrust_design([modules_setting_[1], modules_setting_[1]],
                             [modules_setting_[2], modules_setting_[2]])

    module.set_thrust_bias(noise_[5], noise_[4])
    module.set_control_function(control_set_)
    historical_state_ = module.simulate(5000 * 60, low_step=0.1, progress=False, only_thrust=True, force_step=True,
                                        force_mode=ENTRY_MODE)
    # COST
    mass_state_ = np.array([np.linalg.norm(elem) for elem in historical_state_[2]])
    ang = np.arctan2(historical_state_[0][-1][1], historical_state_[0][-1][0])
    v_t_n = np.array([[np.cos(ang - np.pi / 2), -np.sin(ang - np.pi / 2)],
                      [np.sin(ang - np.pi / 2), np.cos(ang - np.pi / 2)]]).T @ historical_state_[1][-1]
    # tangent velocity
    error = v_t_n[0] ** 2
    return error, historical_state_


def decent_optimization(modules_setting_):
    thruster_pos = np.array([[0.0, -0.0887]
                             # [-0.11, -0.0887], [0.11, -0.0887],
                             # [-0.09, -0.0887], [0.09, -0.0887],
                             # [-0.09, -0.0887], [0.09, -0.0887],
                             # [-0.09, -0.0887], [0.09, -0.0887],
                             # [-0.09, -0.0887], [0.09, -0.0887]
                             ])

    thruster_pos += np.random.normal(0, 0.0001, size=np.shape(thruster_pos))
    thruster_ang = np.random.normal(0, np.deg2rad(0.5), size=(len(thruster_pos)))

    thruster_properties_ = [copy.deepcopy(main_thruster)
                            # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster),
                            # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster),
                            # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster),
                            # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster),
                            # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster)
                            ]

    propellant_properties_ = [copy.deepcopy(main_propellant)
                              # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant),
                              # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant),
                              # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant),
                              # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant),
                              # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant)
                              ]

    state_descent = modules_setting_[1]
    modules_setting_ = modules_setting_[0]

    control_set_ = [modules_setting_[0]
                    # modules_setting_[1], modules_setting_[1],
                    # modules_setting_[2], modules_setting_[2],
                    # modules_setting_[3], modules_setting_[3],
                    # modules_setting_[4], modules_setting_[4],
                    # modules_setting_[5], modules_setting_[5]
                    ]

    state_ = [state_descent[0],
              state_descent[1],
              state_descent[3],
              state_descent[4]
              ]

    sigma_r, sigma_v = 50, 5
    mass_, inertia_ = state_descent[2], state_descent[5]
    module = Module(mass_, inertia_, state_, sigma_r, sigma_v,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)
    module.set_thrust_design([0.15
                              # 0.065, 0.065,
                              # 0.035, 0.035,
                              # 0.035, 0.035,
                              # 0.035, 0.035,
                              # 0.035, 0.035
                              ],
                             [0.2
                              # 0.2, 0.2,
                              # 0.2, 0.2,
                              # 0.2, 0.2,
                              # 0.2, 0.2,
                              # 0.2, 0.2
                              ])

    # module.set_thrust_bias(list(uncertainties["isp"]), dead_time=list(uncertainties["dead"]))
    module.set_control_function(control_set_)
    historical_state_ = module.simulate(5000 * 60, low_step=0.1, progress=False, force_step=False,
                                        force_mode=DESCENT_MODE)

    # COST
    alt_state = np.array([np.linalg.norm(elem) - rm for elem in historical_state_[0]])
    ang = np.arctan2(historical_state_[0][-1][1], historical_state_[0][-1][0])
    v_t_n = np.array([[np.cos(ang - np.pi / 2), -np.sin(ang - np.pi / 2)],
                      [np.sin(ang - np.pi / 2), np.cos(ang - np.pi / 2)]]).T @ historical_state_[1][-1]
    # radial velocity
    error = v_t_n[1] ** 2
    return error, historical_state_


if __name__ == '__main__':
    stage = "E"
    plot_flag = True

    name = name_
    hist_list = []
    dataset = {name: []}
    for j in range(20):
        print(j)
        name_int = name + "_{}".format(j)

        # ============================================================================================================#
        # ENTRY
        # ============================================================================================================#

        range_variables = [(4.5, 5.0),
                           (0.04, 0.04),
                           (0.2, 0.2)]
        n_step = 100
        n_par = 10

        pso_entry = PSOStandard(entry_optimization, n_particles=n_par, n_steps=n_step)
        pso_entry.initialize(range_variables)
        init_time = time.time()
        noise_args = (np.random.normal(0, 0.0001, size=(2, 2)), np.random.normal(0, 0.5, size=2),
                      np.random.normal(0, sigma_r, size=2), np.random.normal(0, sigma_v, size=2),
                      np.random.uniform(0, 0.5, size=2), np.random.normal(0, 10.83, size=2))
        final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_entry.optimize(clip=True, tol=1e-12,
                                                                                                noise=noise_args)
        end_time = time.time()
        print("Optimization Time: {}".format((end_time - init_time) / 60))
        modules_setting_entry = pso_entry.gbest_position

        thruster_pos = np.array([[-0.06975, -0.0887], [0.06975, -0.0887]])
        thruster_pos += noise_args[0]
        thruster_ang = noise_args[1] * np.deg2rad(1)
        thruster_ang += np.array([-25 * np.pi / 180, 25 * np.pi / 180])

        thruster_properties_ = [copy.deepcopy(second_thruster), copy.deepcopy(second_thruster)]
        propellant_properties_ = [copy.deepcopy(second_propellant), copy.deepcopy(second_propellant)]

        r_, v_ = propagate_rv_by_ang(state[0], state[1],
                                     modules_setting_entry[0] - np.deg2rad(0.00),
                                     mu)
        state_ = [r_ + noise_args[2],
                  v_ + noise_args[3],
                  state[2],
                  state[3]]

        module = Module(mass_0, inertia_0, state_, sigma_r, sigma_v,
                        thruster_pos, thruster_ang, thruster_properties_,
                        propellant_properties_, "2D", dt, training=True)

        control_set_ = [modules_setting_entry[0], modules_setting_entry[0]
                        ]
        module.set_control_function(control_set_)

        module.set_thrust_design([modules_setting_entry[1], modules_setting_entry[1]],
                                 [modules_setting_entry[2], modules_setting_entry[2]])
        module.set_thrust_bias(noise_args[5], noise_args[4])
        historical_state = module.simulate(150000 * 60, low_step=0.1, progress=False)

        historical_state.insert(-1, module.historical_theta_error)
        historical_state.insert(-1, module.historical_omega_error)
        historical_state.insert(-1, module.rw_model.historical_rw_velocity)
        historical_state.insert(-1, np.array(module.rw_model.historical_rw_torque) * 1e3)
        hist_list.append(historical_state)

        data = {'state': historical_state,
                'state_name': list_name,
                'best_cost': pso_entry.evol_best_fitness,
                'p_cost': pso_entry.evol_p_fitness,
                'best_part': pso_entry.historical_g_position,
                'hist_part': pso_entry.historical_position
                }
        historical_state[1] = np.array(historical_state[1]) / 1000
        historical_state[7] = np.array(historical_state[7]) * 1000
        historical_state[8] = np.array(historical_state[8]) / 1000
        dataset[name].append(data)
        # ============================================================================================================#
        # DESCENT
        # ============================================================================================================#

        # range_variables = [(3000, 500e3)  # altitude
        #                    # (0, 100e3),
        #                    # (0, 100e3),
        #                    # (0, 100e3),
        #                    # (0, 100e3),
        #                    # (0, 100e3)
        #                    ]
        # n_step = 50
        # n_par = 20
        # pso_descent = PSOStandard(decent_optimization, n_particles=n_par, n_steps=n_step)
        # pso_descent.initialize(range_variables)
        # init_time = time.time()
        # args = [elem[-1] for elem in best_state]
        # final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_descent.optimize(clip=True,
        #                                                                                           args=args)
        # end_time = time.time()
        # print("Optimization Time: {}".format((end_time - init_time) / 60))
        # modules_setting_decent = pso_descent.gbest_position
        #
        # data = {'state': best_state,
        #         'state_name': list_name,
        #         'best_cost': pso_descent.evol_best_fitness,
        #         'p_cost': pso_descent.evol_p_fitness,
        #         'best_part': pso_descent.historical_g_position,
        #         'hist_part': pso_descent.historical_position
        #         }
        #
        # with open(folder + name + "_descent_" + ".pkl", "wb") as data_handle:
        #     pickle.dump(data, data_handle)
        #     data_handle.close()
        #
        #
        # # ============================================================================================================#
        # # FULL SIMULATION
        # # ============================================================================================================#
        #
        # array_thruster_pos = np.array([[-0.11, -0.0887], [0.11, -0.0887],
        #                                [0.0, -0.0887]
        #                                # [-0.11, -0.0887], [0.11, -0.0887],
        #                                # [-0.09, -0.0887], [0.09, -0.0887],
        #                                # [-0.09, -0.0887], [0.09, -0.0887],
        #                                # [-0.09, -0.0887], [0.09, -0.0887],
        #                                # [-0.09, -0.0887], [0.09, -0.0887]
        #                                ])
        #
        # array_thruster_pos += np.random.normal(0, 0.0001, size=np.shape(array_thruster_pos))
        # array_thruster_ang = np.random.normal(0, np.deg2rad(0.5), size=(len(array_thruster_pos)))
        #
        # array_thruster_properties_ = [copy.deepcopy(second_thruster), copy.deepcopy(second_thruster),
        #                               copy.deepcopy(main_thruster),
        #                               # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster),
        #                               # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster),
        #                               # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster),
        #                               # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster),
        #                               # copy.deepcopy(third_thruster), copy.deepcopy(third_thruster)
        #                               ]
        #
        # array_propellant_properties_ = [copy.deepcopy(second_propellant), copy.deepcopy(second_propellant),
        #                                 copy.deepcopy(main_propellant)
        #                                 # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant),
        #                                 # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant),
        #                                 # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant),
        #                                 # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant),
        #                                 # copy.deepcopy(third_propellant), copy.deepcopy(third_propellant)
        #                                 ]
        #
        # state_ = [state[0],
        #           state[1],
        #           state[2],
        #           state[3]]
        #
        # module = Module(mass_0, inertia_0, state_, sigma_r, sigma_v,
        #                 array_thruster_pos, array_thruster_ang, array_thruster_properties_,
        #                 array_propellant_properties_, "2D", dt, training=True)
        #
        # control_set_ = [modules_setting_entry[0], modules_setting_entry[0],
        #                 modules_setting_decent[0],
        #                 # modules_setting_decent[1], modules_setting_decent[1],
        #                 # modules_setting_decent[1], modules_setting_decent[1],
        #                 # modules_setting_decent[1], modules_setting_decent[1],
        #                 # modules_setting_decent[1], modules_setting_decent[1],
        #                 # modules_setting_decent[1], modules_setting_decent[1]
        #                 ]
        # module.set_control_function(control_set_)
        #
        # module.set_thrust_design([0.04, 0.04,
        #                           0.15
        #                           # 0.065, 0.065,
        #                           # 0.035, 0.035,
        #                           # 0.035, 0.035,
        #                           # 0.035, 0.035,
        #                           # 0.035, 0.035
        #                           ],
        #                          [0.2, 0.2,
        #                           0.2
        #                           # 0.2, 0.2,
        #                           # 0.2, 0.2,
        #                           # 0.2, 0.2,
        #                           # 0.2, 0.2,
        #                           # 0.2, 0.2
        #                           ])
        #
        # historical_state = module.simulate(150000 * 60, low_step=0.1, progress=False)
        # state_ = [historical_state[0], historical_state[1]]
        # mass_, inertia_ = historical_state[2][-1], historical_state[5][-1]
        #
        # historical_state.insert(-1, module.historical_theta_error)
        # historical_state.insert(-1, module.historical_omega_error)
        # historical_state.insert(-1, module.rw_model.historical_rw_velocity)
        # historical_state.insert(-1, np.array(module.rw_model.historical_rw_torque) * 1e3)
        # hist_list.append(historical_state)
        #
        # historical_state[1] = np.array(historical_state[1]) / 1000
        # historical_state[7] = np.array(historical_state[7]) * 1000
        # historical_state[8] = np.array(historical_state[8]) / 1000

        # new Agg
        if not plot_flag:
            plt.switch_backend('Agg')
        plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name_int, plot_flag=plot_flag)

    with open(folder + name + "_entry_" + ".pkl", "wb") as data_handle:
        pickle.dump(dataset, data_handle)
        data_handle.close()

    plot_state_solution(hist_list, list_name, folder, name, aux={8: energy_target}, plot_flag=plot_flag)
    plot_normal_tangent_velocity(hist_list, folder, name, plot_flag=plot_flag)
    plot_orbit_solution(hist_list, ["Orbit"], a, b, rp, folder, name, plot_flag=plot_flag)
    # plot_general_solution(hist_list, ["General"], a, b, rp, folder, name, plot_flag=plot_flag)
    plt.show()
