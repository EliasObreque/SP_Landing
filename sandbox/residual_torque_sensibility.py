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

from core.thrust.thruster import Thruster
from core.thrust.thrustProperties import second_thruster
from core.thrust.propellant.propellantProperties import second_propellant

# Thruster
second_thruster['max_ignition_dead_time'] = 0

# Propellant
second_propellant['isp_noise_std'] = 0
second_propellant['isp_bias_std'] = 0

thruster_pos = np.array([[-0.06975, -0.0887], [0.06975, -0.0887]]) + np.random.normal(0, 0.0000, size=2)

thruster_ang = np.random.normal(0, np.deg2rad(0.0), size=(len(thruster_pos)))

thruster_properties_ = [copy.deepcopy(second_thruster), copy.deepcopy(second_thruster)]
propellant_properties_ = [copy.deepcopy(second_propellant), copy.deepcopy(second_propellant)]


if __name__ == '__main__':
    mass_0 = 24.0
    inertia_0 = 1 / 12 * mass_0 * (0.2 ** 2 + 0.3 ** 2)
    dt = 0.01
    n_thruster = 2
    t_end = 20
    dataset = []
    for i in range(1):
        thrusters = [Thruster(dt, thruster_properties_[0], propellant_properties_[0])(),
                     Thruster(dt, thruster_properties_[1], propellant_properties_[1])()]
        print(thrusters[0].throat_diameter, thrusters[0].chamber_pressure)
        time_array = [0]
        hist_torque = [0]
        k = 1
        current_time = 0
        beta = 0
        while current_time <= t_end:
            thr = 0
            if current_time >= 0.5:
                beta = 1
            torque = 0
            for i in range(n_thruster):
                thrusters[i].set_ignition(beta)
                thrusters[i].propagate_thrust()
                thrusters[i].log_value()

                thr_vec = thrusters[i].current_mag_thrust_c * np.array([-np.sin(thruster_ang[i]), np.cos(thruster_ang[i])])
                torque += np.cross(thruster_pos[i], thr_vec)

            current_time += dt
            hist_torque.append(torque)
            time_array.append(current_time)
        total_thrust = [np.array(hist.historical_mag_thrust) for hist in thrusters]

        dataset.append([time_array, total_thrust, hist_torque])

    # plot thrust
    plt.figure()
    plt.grid()
    plt.ylabel("Thrust [N]", fontsize=14)
    plt.xlabel("Time [s]", fontsize=14)
    [plt.plot(data[0], np.array(data[1]).T, color='b') for i, data in enumerate(dataset)]

    # plot torque
    plt.figure()
    plt.grid()
    plt.ylabel("Torque [Nm]", fontsize=14)
    plt.xlabel("Time [s]", fontsize=14)
    [plt.plot(data[0], np.array(data[2]).T, color='b') for i, data in enumerate(dataset)]

    plt.show()
