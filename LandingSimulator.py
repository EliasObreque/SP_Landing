"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from core.module.Module import Module
from core.thrust.thrustProperties import default_thruster
from core.thrust.propellant.propellantProperties import default_propellant
from matplotlib.patches import Ellipse
from tools.pso import PSOStandard, APSO

n_thrusters = 2
thruster_pos = [np.array([0, 0])] * 2
thruster_ang = [0] * 2
thr_properties = default_thruster

# thr_properties['thrust_profile'] = {'type': 'file', 'file_name': 'thrust/dataThrust/5kgEngine.csv', 'isp': 212,
#                                     'dt': 0.01, 'ThrustName': 'Thrust(N)', 'TimeName': 'Time(s)'}

thruster_properties = [thr_properties] * n_thrusters
propellant_properties = [default_propellant] * n_thrusters

reference_frame = '2D'
n_modules = 1
mass_0 = 24.0
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
f_0 = -np.pi/2
r_0 = p_ / (1 + ecc * np.cos(f_0))

rot_90 = np.array([[0, -1], [1, 0]])

position = rot_90 @ np.array([np.cos(f_0), np.sin(f_0)]) * r_0
velocity = np.sqrt(mu/p_) * rot_90 @ np.array([-np.sin(f_0), (ecc + np.cos(f_0))])
theta = -90 * np.deg2rad(1)
q_i2b = np.array([0, 0, 0, 1])
omega = 0
if reference_frame == '2D':
    state = [position, velocity, theta, omega]
elif reference_frame == '3D':
    state = [np.array([*position, 0.0]), np.array([*velocity, 0]), q_i2b, np.zeros(3)]
else:
    assert False

dt = 0.01
tf = 1210000


def get_energy(mu, r, v):
    return 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)


def cost_function(modules_setting, plot=False):
    h_target = rm + 100e3
    rp_target = 2e6
    r_target, v_target, theta_target, omega_target = h_target, np.sqrt(mu * (2/h_target - 1/rp_target)), 0.0, 0.0
    # energy_target = get_energy(mu, r_target, v_target)
    energy_target = -mu / (rp_target + h_target)
    cost = []
    thruster_properties = [thr_properties] * len(modules_setting[1::2])
    propellant_properties = [default_propellant] * len(modules_setting[1::2])
    state_ = []
    energy_module = []
    for i in range(n_modules):
        if reference_frame == '2D':
            st = [state[0] + np.random.normal(0, 100 if n_modules > 1 else 0, size=2),
                  state[1] + np.random.normal(0, 5 if n_modules > 1 else 0, size=2),
                  state[2],
                  state[3]]
        elif reference_frame == '3D':
            st = [state[0] + np.random.normal(0, 100 if n_modules > 1 else 0, size=3),
                  state[1] + np.random.normal(0, 5 if n_modules > 1 else 0, size=3),
                  state[2] + np.random.normal(0, 0.01 if n_modules > 1 else 0, size=4),
                  state[3] + np.random.normal(0, 1e-3 if n_modules > 1 else 0, size=3)]
        else:
            break
        state_.append(st)
    modules_ = [Module(mass_0, inertia_0, state_[i],
                       thruster_pos, thruster_ang, thruster_properties,
                       propellant_properties, reference_frame, dt, training=True) for i in range(n_modules)]
    min_state = []
    for i, module_i in enumerate(modules_):
        engine_diam = modules_setting[1::2]
        control_set = modules_setting[0::2]
        module_i.set_thrust_design(engine_diam, 0)
        module_i.set_control_function(control_set)
        historical_state = module_i.simulate(tf, low_step=0.01, progress=False)
        r_state = np.array([np.linalg.norm(elem) for elem in historical_state[0]])
        v_state = np.array([np.linalg.norm(elem) for elem in historical_state[1]])
        mass_state = np.array([np.linalg.norm(elem) for elem in historical_state[2]])
        state_energy = np.array([get_energy(mu, r_state_, v_state_) for r_state_, v_state_ in zip(r_state, v_state)])
        energy_module.append(state_energy[-1])
        # error = np.abs(state_energy[-1] - energy_target) / mass_state[-1]
        error = ((r_target - r_state[-4]) ** 2 + (v_target - v_state[-1]) ** 2) ** 0.5
        error *= 100 if module_i.dynamics.isTouchdown() else 1
        if module_i.dynamics.notMass():
            error *= 10
        cost.append(error)
        min_state.append(historical_state)
        module_i.reset()
        # cost.append(energy_ite)
    print("cost: {}, energy target: {}, energy: {}, pos: {}, vel: {}".format(
        np.mean(cost), energy_target, np.mean(energy_module), r_state[-1] - rm, v_state[-1]))
    return np.mean(cost), min_state


if __name__ == '__main__':
    current_time = 0

    # Optimal Design of the Control (First stage: Decrease the altitude, and the mass to decrease the rw mass/inertia)
    range_variables = [(0, 2 * np.pi),  # First ignition position (angle)
                       (0.1, 0.2)    # Main engine diameter (meter)
                       #(0, 2 * np.pi),  # Second ignition position (meter)
                       #(0.0, 0.2),  # Secondary engine diameter (meter)
                       ]
    n_step = 50
    n_par = 30
    pso_algorithm = PSOStandard(cost_function, n_particles=n_par, n_steps=n_step)
    pso_algorithm.initialize(range_variables)

    # pso_algorithm_gra = APSO(cost_function, n_particles=n_par, n_steps=n_step)
    # pso_algorithm_gra.range_var = range_variables
    # pso_algorithm_gra.position = pso_algorithm.position.copy()
    # pso_algorithm_gra.velocity = pso_algorithm.velocity.copy()
    # pso_algorithm_gra.pbest_position = pso_algorithm_gra.position
    init_time = time.time()
    final_eval = pso_algorithm.optimize()
    end_time = time.time()
    print("Optimization Time: {}".format((end_time - init_time) / 60))
    modules_setting = pso_algorithm.gbest_position
    pso_algorithm.show_map(0, 1)

    # final_eval_gra = pso_algorithm_gra.optimize()
    # modules_setting_gra = pso_algorithm_gra.gbest_position
    # pso_algorithm_gra.show_map()

    # print("Final evaluation: {}, Final evaluation gra: {}".format(final_eval, final_eval_gra))
    plt.figure()
    plt.plot(pso_algorithm.evol_best_fitness)
    # plt.plot(pso_algorithm_gra.evol_best_fitness, '-o')
    plt.grid()
    plt.legend(['Standard', "apso"])
    plt.yscale("log")
    plt.show()

    # if pso_algorithm_gra.gbest_fitness_value < pso_algorithm.gbest_fitness_value:
    #     modules_setting = modules_setting_gra

    thruster_properties = [thr_properties] * len(modules_setting[1::2])
    propellant_properties = [default_propellant] * len(modules_setting[1::2])
    modules = [Module(mass_0, inertia_0, state, thruster_pos, thruster_ang, thruster_properties,
                      propellant_properties, reference_frame, dt) for i in range(n_modules)]
    for i, module_i in enumerate(modules):
        engine_diam = modules_setting[1::2]
        control_set = modules_setting[0::2]
        module_i.set_thrust_design(engine_diam, 0)
        module_i.set_control_function(control_set)
        final_state = module_i.simulate(tf, low_step=0.1)
        module_i.evaluate()

        print("Final State {}: ".format(final_state))

    plt.figure()
    ax = plt.gca()

    plt.plot(np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 0] * 1e-3,
             np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1] * 1e-3)

    print(modules[0].get_mass_used())

    plt.plot(*modules[0].get_ignition_state('init')[0][0] * 1e-3, 'xg')
    plt.plot(*modules[0].get_ignition_state('end')[0][0] * 1e-3, 'xr')
    plt.plot(*modules[0].get_ignition_state('init')[1][0] * 1e-3, 'xg')
    plt.plot(*modules[0].get_ignition_state('end')[1][0] * 1e-3, 'xr')

    ellipse = Ellipse(xy=(0, -(a - rp) * 1e-3), width=b * 2 * 1e-3, height=2 * a * 1e-3,
                      edgecolor='r', fc='None', lw=0.7)
    ellipse_moon = Ellipse(xy=(0, 0), width=2 * rm * 1e-3, height=2 * rm * 1e-3, fill=True,
                           edgecolor='black', fc='gray', lw=0.4)
    ax.add_patch(ellipse)
    ax.add_patch(ellipse_moon)
    plt.grid()

    plt.figure()
    plt.title("Y-Position")

    plt.plot(modules[0].dynamics.dynamic_model.historical_time,
             np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1] * 1e-3)
    plt.plot(modules[0].dynamics.dynamic_model.historical_time,
             np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1] * 1e-3, '+')

    plt.plot(modules[0].get_ignition_state('init')[0][-1], modules[0].get_ignition_state('init')[0][0][1] * 1e-3, 'xg')
    plt.plot(modules[0].get_ignition_state('end')[0][-1], modules[0].get_ignition_state('end')[0][0][1] * 1e-3, 'xr')
    plt.grid()

    plt.figure()
    plt.title("X-Position")
    plt.plot(modules[0].dynamics.dynamic_model.historical_time,
             np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 0] * 1e-3)

    plt.plot(modules[0].get_ignition_state('init')[0][-1], modules[0].get_ignition_state('init')[0][0][0] * 1e-3, 'xg')
    plt.plot(modules[0].get_ignition_state('end')[0][-1], modules[0].get_ignition_state('end')[0][0][0] * 1e-3, 'xr')
    plt.grid()

    plt.figure()
    plt.plot(modules[0].dynamics.dynamic_model.historical_time,
             np.array(modules[0].dynamics.dynamic_model.historical_qi2b))
    plt.grid()

    plt.figure()
    plt.title("Mass (kg)")
    plt.plot(modules[0].dynamics.dynamic_model.historical_time,
             np.array(modules[0].dynamics.dynamic_model.historical_mass))
    plt.plot(modules[0].dynamics.dynamic_model.historical_time,
             np.array(modules[0].dynamics.dynamic_model.historical_mass), '+')
    plt.grid()

    plt.figure()
    plt.title("Thrust (N)")
    plt.plot(modules[0].thrusters[0].get_time(), modules[0].thrusters[0].historical_mag_thrust)
    plt.plot(modules[0].thrusters[0].get_time(), modules[0].thrusters[0].historical_mag_thrust, '+')
    plt.plot(modules[0].thrusters[1].get_time(), modules[0].thrusters[1].historical_mag_thrust)
    plt.plot(modules[0].thrusters[1].get_time(), modules[0].thrusters[1].historical_mag_thrust, '+')

    plt.plot(modules[0].get_ignition_state('init')[0][-1],
             modules[0].thrusters[0].historical_mag_thrust[modules[0].thrusters_action_wind[0][0]], 'xg')
    plt.plot(modules[0].get_ignition_state('end')[0][-1],
             modules[0].thrusters[0].historical_mag_thrust[modules[0].thrusters_action_wind[0][1]], 'xr')

    plt.grid()
    plt.show()
