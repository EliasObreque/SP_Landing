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
from core.thrust.thrustProperties import default_thruster
from core.thrust.propellant.propellantProperties import default_propellant
from tools.pso import PSOStandard
from tools.Viewer import plot_orbit_solution, plot_state_solution, plot_pso_result

mass_0 = np.array(24.0)
inertia_0 = 100
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

# TARGET
h_target = rm + 100e3
rp_target = 2e6
energy_target = [-mu / (rp_target + h_target),
                 -mu / (2 * h_target)]


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
    tf = 1210000

    thruster_pos = [np.array([0, 0]),
                    np.array([0, 0])]
    thruster_ang = [0.0, 0.0]

    thruster_properties_ = [copy.deepcopy(default_thruster) for _ in range(len(modules_setting_[1::2]))]
    propellant_properties_ = [copy.deepcopy(default_propellant) for _ in range(len(modules_setting_[1::2]))]

    thruster_properties_[1]['throat_diameter'] = 0.005
    propellant_properties_[1]['geometry']['setting']['int_diameter'] = 0.01

    state_ = [state[0] + np.random.normal(0, 100, size=2),
              state[1] + np.random.normal(0, 5, size=2),
              state[2],
              state[3]]
    module = Module(mass_0, inertia_0, state_,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)

    engine_diameter = modules_setting_[1::2]
    control_set_ = modules_setting_[0::2]
    module.set_thrust_design(engine_diameter, 0)
    module.set_control_function(control_set_)
    historical_state = module.simulate(tf, low_step=0.01, progress=False)
    r_state = np.array([np.linalg.norm(elem) for elem in historical_state[0]])
    mass_state = np.array([np.linalg.norm(elem) for elem in historical_state[2]])
    state_energy = historical_state[8]
    error = 0.0
    for j, act in enumerate(module.thrusters_action_wind):
        if len(act) > 0:
            error += np.abs(state_energy[min(act[1] + 1, len(state_energy) - 5)] - energy_target[j]) / mass_state[-1]
        else:
            error += np.abs(state_energy[-1] - energy_target[j]) / mass_state[-1]
    error *= 1000 if module.dynamics.isTouchdown() else 1
    error *= 100 if module.dynamics.notMass() else 1
    error *= 1000 if min(r_state) < h_target * 0.9 else 1

    module.reset()
    return error, historical_state


if __name__ == '__main__':
    # Fix multiprocessing with freeze (pyinstaller) on Windows
    # REF: https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
    multiprocessing.freeze_support()

    # python .\LandingSimulator.py -f regressive -n block1 -bs 10 -l 2 -s D -ps 0
    n_step = 150
    n_par = 30
    folder = "logs/"
    name = "test"
    stage = "D"
    plot_flag = False

    batch_size = 3
    n_loop = 2

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
                range_variables = [(1.0, 2),  # First ignition position (angle)
                                   (0.1, 0.19),  # Main engine diameter (meter)
                                   (3, 6),  # Second ignition position (meter)
                                   (0.001, 0.1)  # Secondary engine diameter (meter)
                                   ]

                pso_algorithm = PSOStandard(cost_function_descend, n_particles=n_par, n_steps=n_step)
                pso_algorithm.initialize(range_variables)

                init_time = time.time()
                final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_algorithm.optimize()
                end_time = time.time()
                print("Optimization Time: {}".format((end_time - init_time) / 60))
                plt.switch_backend('Agg')
                plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name_temp, plot_flag=plot_flag)

                list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
                             "Inertia [kgm2]", "Thrust [N]", "Torque [Nm]", "Energy [J]"]
                plot_state_solution(best_state, list_name, folder, name_temp, aux={8: energy_target},
                                    plot_flag=plot_flag)
                plot_orbit_solution([best_state], ["orbit"], folder, name_temp, plot_flag=plot_flag)
                modules_setting = pso_algorithm.gbest_position
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
