"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import time
import copy
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import numpy as np
import datetime
import multiprocessing

from core.module.Module import Module
from core.thrust.thrustProperties import main_thruster, second_thruster, third_thruster
from core.thrust.propellant.propellantProperties import main_propellant, second_propellant, third_propellant
from tools.pso import PSOStandard
from tools.Viewer import plot_orbit_solution, plot_state_solution, plot_pso_result
from tools.mathtools import propagate_rv_by_ang

mass_0 = 24.0
inertia_0 = 1 / 12 * mass_0 * (0.2 ** 2 + 0.3 ** 2)

mu = 4.9048695e12  # m3s-2
rm = 1.738e6
ra = 2e6 # 68e6
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

# TARGET
h_target = rm + 2e3
rp_target = 2e6

energy_target = -mu / h_target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, help="Folder name")
    parser.add_argument("-n", "--name", type=str, help="Files head name")
    parser.add_argument("-bs", "--batch", type=int, help="Batch Size")
    parser.add_argument("-l", "--loop", type=int, help="Number of loops")
    parser.add_argument("-s", "--stage", type=str, help="Stage, Descend (D), Landing (L)")
    parser.add_argument("-ps", "--plot_show", type=int, help="Show")
    return parser.parse_args()


def get_energy(mu, r, v):
    return 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)


def cost_function_descend(modules_setting_):
    state = [position, velocity, theta, omega]
    dt = 0.1
    tf = min(50000, 2 * np.pi / np.sqrt(mu) * a ** (3/2))

    thruster_pos = np.array([[-0.06975, -0.0],  # Main Thruster
                             [-0.06975, -0.0887],  # Second -1- Thruster
                             [-0.06975, 0.0887],  # Second -2- Thruster
                             [-0.06975, -0.0887],  # Second -3- Thruster
                             [-0.06975, 0.0887],  # Second -4- Thruster
                             # [-0.06975, 0.0887],  # Third -1- Thruster
                             # [-0.06975, 0.0887],  # Third -2- Thruster
                             # [-0.06975, 0.0887],  # Third -3- Thruster
                             # [-0.06975, 0.0887],  # Third -4- Thruster
                             # [-0.06975, 0.0887],  # Third -5- Thruster
                             # [-0.06975, 0.0887],  # Third -6- Thruster
                             # [-0.06975, 0.0887],  # Third -7- Thruster
                             # [-0.06975, 0.0887],  # Third -8- Thruster
                             # [-0.06975, 0.0887],  # Third -9- Thruster
                             # [-0.06975, 0.0887],  # Third -10- Thruster
                             # [-0.06975, 0.0887],  # Third -11- Thruster
                             # [-0.06975, 0.0887],  # Third -12- Thruster
                             ])

    thruster_pos += np.random.normal(0, 0.0001, size=np.shape(thruster_pos))

    thruster_ang = np.zeros(len(thruster_pos))

    thruster_ang += np.random.normal(0, np.deg2rad(0.5), size=(len(thruster_ang)))

    thruster_properties_ = [copy.deepcopy(main_thruster),  # Main Thruster
                            copy.deepcopy(second_thruster),  # Second -1- Thruster
                            copy.deepcopy(second_thruster),  # Second -2- Thruster
                            copy.deepcopy(second_thruster),  # Second -3- Thruster
                            copy.deepcopy(second_thruster),  # Second -4- Thruster
                            # copy.deepcopy(third_thruster),  # Third -1- Thruster
                            # copy.deepcopy(third_thruster),  # Third -2- Thruster
                            # copy.deepcopy(third_thruster),  # Third -3- Thruster
                            # copy.deepcopy(third_thruster),  # Third -4- Thruster
                            # copy.deepcopy(third_thruster),  # Third -5- Thruster
                            # copy.deepcopy(third_thruster),  # Third -6- Thruster
                            # copy.deepcopy(third_thruster),  # Third -7- Thruster
                            # copy.deepcopy(third_thruster),  # Third -8- Thruster
                            # copy.deepcopy(third_thruster),  # Third -9- Thruster
                            # copy.deepcopy(third_thruster),  # Third -10- Thruster
                            # copy.deepcopy(third_thruster),  # Third -11- Thruster
                            # copy.deepcopy(third_thruster),  # Third -12- Thruster
                            ]

    propellant_properties_ = [copy.deepcopy(main_propellant),   # Main Thruster
                              copy.deepcopy(second_propellant),  # Second -1- Thruster
                              copy.deepcopy(second_propellant),  # Second -2- Thruster
                              copy.deepcopy(second_propellant),  # Second -3- Thruster
                              copy.deepcopy(second_propellant),  # Second -4- Thruster
                              # copy.deepcopy(third_propellant),  # Third -1- Thruster
                              # copy.deepcopy(third_propellant),  # Third -2- Thruster
                              # copy.deepcopy(third_propellant),  # Third -3- Thruster
                              # copy.deepcopy(third_propellant),  # Third -4- Thruster
                              # copy.deepcopy(third_propellant),  # Third -5- Thruster
                              # copy.deepcopy(third_propellant),  # Third -6- Thruster
                              # copy.deepcopy(third_propellant),  # Third -7- Thruster
                              # copy.deepcopy(third_propellant),  # Third -8- Thruster
                              # copy.deepcopy(third_propellant),  # Third -9- Thruster
                              # copy.deepcopy(third_propellant),  # Third -10- Thruster
                              # copy.deepcopy(third_propellant),  # Third -11- Thruster
                              # copy.deepcopy(third_propellant),  # Third -12- Thruster
                              ]

    state_ = [state[0] + np.random.normal(0, 1e3, size=2),
              state[1] + np.random.normal(0, 5, size=2),
              state[2],
              state[3]]

    # transform cartessian r and v into perifocal r and v
    state_[0], state_[1] = propagate_rv_by_ang(state_[0], state_[1], modules_setting_[1::2][0], mu, ecc)

    module = Module(mass_0, inertia_0, state_,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)

    # PSO solution
    engine_diameter = [modules_setting_[1::2][0],
                       modules_setting_[1::2][1],
                       modules_setting_[1::2][1],
                       modules_setting_[1::2][2],
                       modules_setting_[1::2][2]
                       ]

    control_set_ = [modules_setting_[0],
                    modules_setting_[1], modules_setting_[1],
                    modules_setting_[2], modules_setting_[2],
                    # modules_setting_[3], modules_setting_[3],
                    # modules_setting_[4], modules_setting_[4],
                    # modules_setting_[5], modules_setting_[5],
                    # modules_setting_[6], modules_setting_[6],
                    # modules_setting_[7], modules_setting_[7],
                    # modules_setting_[8], modules_setting_[8],
                    ]

    module.set_thrust_design(engine_diameter, 0)
    module.set_control_function(control_set_)
    historical_state = module.simulate(tf, low_step=0.01, progress=False)
    r_state = np.array([np.linalg.norm(elem) for elem in historical_state[0]])
    v_state = np.array([np.linalg.norm(elem) for elem in historical_state[1]])
    mass_state = np.array([np.linalg.norm(elem) for elem in historical_state[2]])
    state_energy = historical_state[8]
    error = (state_energy[-1] - energy_target) ** 2 / mass_state[-1]

    # ang = np.arctan2(historical_state[0][-1][1], historical_state[0][-1][0])
    # v_t_n = np.array([[np.cos(ang - np.pi/2), -np.sin(ang - np.pi/2)],
    #                   [np.sin(ang - np.pi/2), np.cos(ang - np.pi/2)]]).T @ historical_state[1][-1]
    #
    # error = abs((r_state[-1] - rm)) + 100 * abs(v_t_n[0]) + abs(v_t_n[1])
    # error = state_energy[-1]
    # for j, act in enumerate(module.thrusters_action_wind):
    #     if len(act) > 0:
    #         error += np.abs(state_energy[min(act[1] + 1, len(state_energy) - 5)] - energy_target[j]) / mass_state[-1]
    #     else:
    #         error += np.abs(state_energy[-1] - energy_target[j]) / mass_state[-1]

    error *= 1000 if module.dynamics.isTouchdown() else 1
    error *= 100 if module.dynamics.notMass() else 1
    # error *= 1000 if abs(min(r_state) - h_target) > 2e3 else 1

    module.reset()
    return error, historical_state


if __name__ == '__main__':
    # python .\LandingSimulator.py -f regressive -n block1 -bs 10 -l 2 -s D -ps 0
    n_step = 20
    n_par = 20
    folder = "logs/"
    name = "second_ignition_3"
    stage = "D"
    plot_flag = True

    batch_size = 1
    n_loop = 1

    args = parse_args()

    if args.folder:
        date_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
        folder = folder + args.folder + "_{}_".format(stage) + date_name + "/"
        # create folder
        if not os.path.exists(os.path.normpath(folder)):
            os.makedirs(os.path.normpath(folder))
    else:
        folder = folder + "plane/"

    if args.name:
        name = args.name

    if args.batch:
        batch_size = args.batch

    if args.loop:
        n_loop = args.loop

    if args.stage:
        stage = args.stage
    if args.plot_show:
        plot_flag = args.plot_show

    for nl in range(n_loop):
        data_handle = open(folder + name + "_{}".format(nl), 'wb')
        print(folder + name + "_{}".format(nl))
        dataset = {}
        for nb in range(batch_size):
            name_temp = name + "_{}".format(nb + nl * batch_size)
            print(name_temp)
            if stage == "D":
                # Optimal Design of the Control
                # (First stage: Decrease the altitude, and the mass to decrease the rw mass/inertia)
                range_variables = [(0.0, np.pi),  # First ignition position (angle)
                                   (0.1, 0.15),  # Main engine diameter (meter)
                                   (np.pi/2, 2 * np.pi),  # Second ignition position (meter)
                                   (0.03, 0.06),  # Secondary engine diameter (meter)
                                   (np.pi/2, 2 * np.pi),  # 3 ignition position (meter)
                                   (0.03, 0.06)  # 3 engine diameter (meter)
                                   ]
                # range_variables = [(1.4, 3),  # First ignition angular (rad)
                #                    (4, 1.7 * np.pi),  # Second -1- ignition angular (rad)
                #                    (4, 1.7 * np.pi),  # Second -2- ignition angular (rad)
                #                    # (0, 50e3),  # Third -1- ignition position (meter)
                #                    # (0, 50e3),  # Third -2- ignition position (meter)
                #                    # (0, 30e3),  # Third -3- ignition position (meter)
                #                    # (0, 30e3),  # Third -4- ignition position (meter)
                #                    # (0, 10e3),  # Third -5- ignition position (meter)
                #                    # (0, 10e3),  # Third -6- ignition position (meter)
                #                    ]
                pso_algorithm = PSOStandard(cost_function_descend, n_particles=n_par, n_steps=n_step)
                pso_algorithm.initialize(range_variables)

                init_time = time.time()
                final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_algorithm.optimize(clip=True)
                end_time = time.time()
                print("Optimization Time: {}".format((end_time - init_time) / 60))
                modules_setting = pso_algorithm.gbest_position

                list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
                             "Inertia [kgm2]", "Thrust [N]", "Torque [Nm]", "Energy [J]"]
                if plot_flag:
                    plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name_temp, plot_flag=plot_flag)
                    plot_state_solution(best_state, list_name, folder, name_temp, aux={8: energy_target},
                                        plot_flag=plot_flag)
                    plot_orbit_solution([best_state], ["orbit"], a, b, rp, folder, name_temp,
                                        h_target=h_target, plot_flag=plot_flag)
                    plt.show(block=True)
            elif stage == "L":
                pass

            data = {'state': best_state,
                    'state_name': list_name,
                    'best_cost': pso_algorithm.evol_best_fitness,
                    'p_cost': pso_algorithm.evol_p_fitness,
                    'best_part': pso_algorithm.historical_g_position,
                    'hist_part': pso_algorithm.historical_position}

            dataset[name_temp] = data
            if plot_flag:
                plt.show(block=False)
        pickle.dump(dataset, data_handle)
        data_handle.close()
