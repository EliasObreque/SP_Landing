"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
import matplotlib.pyplot as plt
from module.Module import Module
import thrust.thrustProperties as thr_prop
import thrust.propellant.propellantProperties as prop_prop
from matplotlib.patches import Ellipse

n_thrusters = 1
thruster_pos = [np.array([0, 0])]
thruster_ang = [0]
thr_properties = thr_prop.default_thruster

# thr_properties['thrust_profile'] = {'type': 'file', 'file_name': 'thrust/dataThrust/5kgEngine.csv', 'isp': 212,
#                                     'dt': 0.01, 'ThrustName': 'Thrust(N)', 'TimeName': 'Time(s)'}

thruster_properties = [thr_properties] * n_thrusters
propellant_properties = [prop_prop.default_propellant] * n_thrusters

reference_frame = '2D'
n_modules = 1
mass_0 = 24.0
inertia_0 = 100
mu = 4.9048695e12  # m3s-2
rm = 1.738e6
ra = 68e6
rp = 2e6
a = 0.5 * (ra + rp)
ecc = 1 - rp/a
b = a * np.sqrt(1 - ecc**2)
vp = np.sqrt(2 * mu / rp - mu / a)
va = np.sqrt(2 * mu / ra - mu / a)
position = np.array([0, rp])
velocity = np.array([-vp, 0])
theta = -90*np.deg2rad(1)
omega = 0
state = [position, velocity, theta, omega]

current_time = 0
dt = 0.01
tf = 5*60*60
modules = [Module(mass_0, inertia_0, state, n_thrusters, thruster_pos, thruster_ang, thruster_properties,
                  propellant_properties, reference_frame, dt) for i in range(n_modules)]

# Optimal Design of the Control
control = [0] * len(modules)
k = 0
subk = 0
wind_thrust = [[]] * len(modules)
while current_time <= tf:
    for i, module_i in enumerate(modules):
        if current_time >= 0 and module_i.thrusters[i].thr_is_burned == False:
            control[i] = 1
        else:
            control[i] = 0
        if abs(current_time - module_i.dynamics.dynamic_model.current_time) < 0.5 * dt or current_time >= module_i.dynamics.dynamic_model.current_time:
            if control[i] == 1 and module_i.thrusters[i].current_dead_time > 0 and module_i.thrusters[i].thr_is_burned == False:
                wind_thrust[i].append(subk) if len(wind_thrust[i]) == 0 else None
                low_step = True
            else:
                wind_thrust[i].append(subk) if len(wind_thrust[i]) == 1 else None
                low_step = False
            module_i.update(control[-1], low_step=low_step, dt=dt)
            module_i.dynamics.dynamic_model.save_data()
            subk += 1
    current_time += dt
    if k > 29:
        print(current_time / tf * 100, "%")
        k = 0
    k += 1

plt.figure()
ax = plt.gca()
ellipse = Ellipse(xy=(0, -(a - rp) * 1e-3), width=b * 2 * 1e-3, height=2 * a * 1e-3,
                        edgecolor='r', fc='None', lw=0.7)
ellipse_moon = Ellipse(xy=(0, 0), width=2 * rm * 1e-3, height=2 * rm * 1e-3,
                       edgecolor='gray', fc='None', lw=0.7)
plt.plot(np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 0] * 1e-3,
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1] * 1e-3)
plt.plot(np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 0][wind_thrust[0][0]] * 1e-3,
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1][wind_thrust[0][0]] * 1e-3, 'xg')
plt.plot(np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 0][wind_thrust[0][1]] * 1e-3,
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1][wind_thrust[0][1]] * 1e-3, 'xr')


ax.add_patch(ellipse)
ax.add_patch(ellipse_moon)
plt.grid()

plt.figure()
plt.title("Y-Position")
plt.plot(modules[0].dynamics.dynamic_model.historical_time,
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1] * 1e-3)
plt.plot(modules[0].dynamics.dynamic_model.historical_time[wind_thrust[0][0]],
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1][wind_thrust[0][0]] * 1e-3, 'xg')
plt.plot(modules[0].dynamics.dynamic_model.historical_time[wind_thrust[0][1]],
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 1][wind_thrust[0][1]] * 1e-3, 'xr')
plt.grid()

plt.figure()
plt.title("X-Position")
plt.plot(modules[0].dynamics.dynamic_model.historical_time,
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 0] * 1e-3)
plt.plot(modules[0].dynamics.dynamic_model.historical_time[wind_thrust[0][0]],
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 0][wind_thrust[0][0]] * 1e-3, 'xg')
plt.plot(modules[0].dynamics.dynamic_model.historical_time[wind_thrust[0][1]],
         np.array(modules[0].dynamics.dynamic_model.historical_pos_i)[:, 0][wind_thrust[0][1]] * 1e-3, 'xr')
plt.grid()

plt.figure()
plt.plot(modules[0].dynamics.dynamic_model.historical_time,
         np.array(modules[0].dynamics.dynamic_model.historical_theta))
plt.grid()

plt.figure()
plt.title("Mass (kg)")
plt.plot(modules[0].dynamics.dynamic_model.historical_time,
         np.array(modules[0].dynamics.dynamic_model.historical_mass))
plt.grid()
plt.xlim([0, 15])

plt.figure()
plt.title("Thrust (N)")
plt.plot(modules[0].dynamics.dynamic_model.historical_time, modules[0].thrusters[0].historical_mag_thrust)
plt.plot(modules[0].dynamics.dynamic_model.historical_time[wind_thrust[0][0]],
         modules[0].thrusters[0].historical_mag_thrust[wind_thrust[0][1]], 'xg')
plt.plot(modules[0].dynamics.dynamic_model.historical_time[wind_thrust[0][1]],
         modules[0].thrusters[0].historical_mag_thrust[wind_thrust[0][1]], 'xr')
plt.grid()
plt.xlim([0, 10])
plt.show()



