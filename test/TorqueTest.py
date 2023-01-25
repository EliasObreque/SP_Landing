"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 04-08-2022
"""
from thrust.propellant.propellant import propellant_data
from thrust.thruster import Thruster
import numpy as np
import matplotlib.pyplot as plt


TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'

NEUTRAL = 'neutral'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'

m0 = 24
propellant_name = 'CDT(80)'
selected_propellant = [pro_data for pro_data in propellant_data if pro_data['name'] == propellant_name][0]
propellant_geometry = TUBULAR
Isp = selected_propellant['data']['Isp']
den_p = selected_propellant['data']['density']
ge = 9.807
c_char = Isp * ge
g_center_body = -1.62
r_moon = 1738e3
mu = 4.9048695e12
reference_frame = '1D'
dt = 0.01

engine_diameter_ext = None
throat_diameter = 1.0  # mm
height = 10.0  # mm
file_name = "thrust/StarGrain7.csv"

propellant_properties_ = {'propellant_name': propellant_name,
                          'n_thrusters': 1,
                          'pulse_thruster': 1,
                          'geometry': None,
                          'propellant_geometry': propellant_geometry,
                          'isp_noise_std': None,
                          'isp_bias_std': None,
                          'isp_dead_time_max': 0.0}

ctrl_a = [1.0]
ctrl_b = [6.91036]
optimal_alpha = 10.16e-3 / 2
t_burn = 20.0
json_list = {'1': {'Best_individual': [optimal_alpha, t_burn, ctrl_a, ctrl_b]}}

percentage_variation = 3
upper_isp = Isp * (1.0 + percentage_variation / 100.0)
propellant_properties_['isp_noise_std'] = (upper_isp - Isp) / 3

percentage_variation = 10
upper_isp = Isp * (1.0 + percentage_variation / 100.0)
propellant_properties_['isp_bias_std'] = (upper_isp - Isp) / 3

dead_time = 1
lag_coef = 0.5
thruster_properties_ = {'throat_diameter': 2,
                        'engine_diameter_ext': engine_diameter_ext,
                        'height': height,
                        'performance': {'alpha': optimal_alpha,
                                        't_burn': t_burn},
                        'load_thrust_profile': False,
                        'file_name': file_name,
                        'dead_time': 0.2,
                        'lag_coef': lag_coef}

n_thruster = 10
comp_thrust = []
for i in range(n_thruster):
    propellant_properties_['isp_dead_time_max'] = 0.5
    comp_thrust.append(Thruster(dt, thruster_properties_, propellant_properties_))


# Torque
J = ((m0 - 10) / 12) * (0.2 ** 2 + 0.3 ** 2)
print("Inertia: ", J)
sigma_d = 2 * 1e-3  # mm
sigma_a = np.deg2rad(2.0)
d0 = 0.08
d = [d0, -d0] * n_thruster
color_arra = ['g', 'r'] * n_thruster
d_k = []
a_k = []
for i in range(n_thruster):
    d_k.append(d[i] + np.random.normal(0, sigma_d))
    a_k.append(np.random.normal(0, sigma_a))

kappa = 0.2

control = []

fc = 0.006
Jrw = J * fc
print(Jrw)
r_rw = np.arange(0.05, 0.11, 0.01) / 2
m = Jrw * 2 / r_rw ** 2
print(m)


def dynamics_1d(x, torque_i):
    dx = np.zeros(5)
    dx[0] = x[1]
    dx[1] = torque_i / J

    acc_ctrl = torque_i / Jrw
    target_ang_vel = x[2] + acc_ctrl * dt + x[3] * 200 + x[4] * 50
    dx[2] = (target_ang_vel - x[2])/kappa

    dx[3] = x[4]
    dx[4] = (torque_i - dx[2] * Jrw) / J
    return dx


def rungeonestep(state, thrust):
    # At each step of the Runge consider constant thrust
    x_fixed = np.array(state)
    x1 = x_fixed
    k1 = dynamics_1d(x1, thrust)
    xk2 = x1 + (dt / 2.0) * k1
    k2 = dynamics_1d(xk2, thrust)
    xk3 = x1 + (dt / 2.0) * k2
    k3 = dynamics_1d(xk3, thrust)
    xk4 = x1 + dt * k3
    k4 = dynamics_1d(xk4, thrust)
    next_x = x1 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_x


time_array = []
k = 1
current_time = 0.0
x0 = np.array([0, 0, 0, 0, 0])
X = [x0]
torque_array = []
torque_2 = 0
beta = [0] * int(n_thruster)
counter = 0
while current_time <= t_burn * n_thruster * 0.5 * 1.1:
    time_array.append(current_time)
    thr = 0
    for i in range(int(n_thruster/2)):
        if 1 + t_burn * (i + 1) > current_time > 1 + t_burn * i:
            if X[-1][3] < -1:
                beta[i * 2] = 1
            elif X[-1][3] > 1:
                beta[i * 2 + 1] = 1
            else:
                beta[i * 2] = 1
                beta[i * 2 + 1] = 1
                break
            if counter % 0.1 == 0:
                if X[-1][3] < -1:
                    beta[i * 2 + 1] = 1
                    counter = 0
                elif X[-1][3] > 1:
                    beta[i * 2] = 1
                    counter = 0
            counter += dt
            break
    torque_i = 0
    for i in range(n_thruster):
        comp_thrust[i].set_ignition(beta[i])
        comp_thrust[i].propagate_thrust()
        comp_thrust[i].log_value()
        torque_i += comp_thrust[i].get_current_thrust() * d_k[i] * np.sin(np.pi/2 + a_k[i])

    new_x = rungeonestep(X[-1], torque_i)
    X.append(new_x)
    current_time += dt

total_thrust = 0
torque_array = []
i = 0
torque = 0
for hist in comp_thrust:
    torque += np.array(hist.historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i])
    total_thrust += np.array(hist.historical_mag_thrust)
    i += 1
    if i % 2 == 0:
        torque_array.append(torque)
        torque = 0


plt.figure()
plt.xlabel('Time [s]')
plt.ylabel('thrust [N]')
plt.plot(time_array, total_thrust)
[plt.plot(time_array, thrust.historical_mag_thrust) for thrust in comp_thrust]
plt.grid()
plt.tight_layout()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel('Torque [mNm]')
[plt.plot(time_array, np.array(torque_array[i])*1e3, lw=1) for i in range(int(len(comp_thrust)/2))]
# [plt.plot(time_array, np.array(comp_thrust[i].historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i]),
#           color=color_arra[i], lw=0.7) for i in range(len(comp_thrust))]
plt.grid()
plt.tight_layout()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta$ [deg]')
plt.plot(time_array, np.array(X)[:-1, 0] * 180/np.pi, lw=1)
#[plt.plot(time_array, np.array(comp_thrust[i].historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i]),
#          color=color_arra[i], lw=0.7) for i in range(len(comp_thrust))]
plt.grid()
plt.tight_layout()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel(r'$\omega_b^i$ [deg/s]')
plt.plot(time_array, np.array(X)[:-1, 1] * 180/np.pi, lw=1)
# [plt.plot(time_array, np.array(comp_thrust[i].historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i]),
#           color=color_arra[i], lw=0.7) for i in range(len(comp_thrust))]
plt.grid()
plt.tight_layout()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel(r'$\omega_{RW}^b$ [deg/s]')
plt.plot(time_array, np.array(X)[:-1, 2] * 180/np.pi, lw=1)
# [plt.plot(time_array, np.array(comp_thrust[i].historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i]),
#           color=color_arra[i], lw=0.7) for i in range(len(comp_thrust))]
plt.grid()
plt.tight_layout()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel(r'$H_{RW}^b$ [Nms]')
plt.plot(time_array, np.array(X)[:-1, 2] * Jrw, lw=1)
# [plt.plot(time_array, np.array(comp_thrust[i].historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i]),
#           color=color_arra[i], lw=0.7) for i in range(len(comp_thrust))]
plt.grid()
plt.tight_layout()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel(r'$\tau_{RW}^b$ [mNm]')
plt.plot(time_array, np.diff(np.array(X)[:, 2]) * Jrw * 1e3 / dt, lw=1)
# [plt.plot(time_array, np.array(comp_thrust[i].historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i]),
#           color=color_arra[i], lw=0.7) for i in range(len(comp_thrust))]
plt.grid()
plt.tight_layout()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta_c$ [deg]')
plt.plot(time_array, np.array(X)[:-1, 3] * 180/np.pi, lw=1)
# [plt.plot(time_array, np.array(comp_thrust[i].historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i]),
#           color=color_arra[i], lw=0.7) for i in range(len(comp_thrust))]
plt.grid()
plt.tight_layout()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel(r'$\omega_c$ [deg/s]')
plt.plot(time_array, np.array(X)[:-1, 4] * 180/np.pi, lw=1)
# [plt.plot(time_array, np.array(comp_thrust[i].historical_mag_thrust) * d_k[i] * np.sin(np.pi/2 + a_k[i]),
#           color=color_arra[i], lw=0.7) for i in range(len(comp_thrust))]
plt.grid()
plt.tight_layout()

plt.show()
