"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
import matplotlib.pyplot as plt
from core.module.Module import Module
from core.thrust.thrustProperties import default_thruster
from core.thrust.propellant.propellantProperties import default_propellant
from matplotlib.patches import Ellipse
from tools.pso import PSORegression

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
position = np.array([0, rp])
velocity = np.array([-vp, 0])
theta = -90 * np.deg2rad(1)
omega = 0
state = [position, velocity, theta, omega]


current_time = 0
dt = 0.1
tf = 180 * 60 * 60
modules = [Module(mass_0, inertia_0, state, thruster_pos, thruster_ang, thruster_properties,
                  propellant_properties, reference_frame, dt) for i in range(n_modules)]


def get_energy(mu, r, v):
    return 0.5 * np.linalg.norm(v) ** 2 - mu/np.linalg.norm(r)


def cost_function(modules_setting, plot=False):
    h_target = rm + 1000e3
    r_target, v_target, theta_target, omega_target = h_target, np.sqrt(mu/h_target), 0.0, 0.0
    energy_target = get_energy(mu, r_target, v_target)
    min_state = []
    min_cost = np.inf
    cost = []
    for i, module_i in enumerate(modules):
        module_i.set_thrust_design([modules_setting[1], modules_setting[3]], 0)
        module_i.set_control_function([modules_setting[0], modules_setting[2]])
        historical_state = module_i.simulate(tf, low_step=0.01, progress=False)
        states = module_i.dynamics.get_current_state()
        module_i.reset()

        r_state, v_state = np.linalg.norm(states[0]), np.linalg.norm(states[1])
        #error = np.abs(get_energy(mu, r_state, v_state) - energy_target)
        error = (0.1 * (r_target - r_state) ** 2 + (v_target - v_state) ** 2) ** 0.5
        cost.append(error)
        if error < min_cost:
            min_state = historical_state
            min_cost = error
        # cost.append(energy_ite)
    print(np.mean(cost))
    return np.mean(cost), min_state


# Optimal Design of the Control (First stage: Decrease the altitude, and the mass to decrease the rw mass/inertia)
range_variables = [(0, 2 * np.pi),  # First ignition position (angle)
                   (0.0, 0.7),    # Main engine diameter (meter)
                   (0, 2 * np.pi),  # Second ignition position (meter)
                   (0, 0.1),  # Secondary engine diameter (meter)
                   ]

pso_algorithm = PSORegression(cost_function, n_particles=100, n_steps=100)
pso_algorithm.initialize(range_variables)
final_eval = pso_algorithm.optimize()
modules_setting = pso_algorithm.gbest_position

plt.figure()
plt.plot(pso_algorithm.historical_loss)
plt.grid()

print("Final evaluation: {}".format(final_eval))

for i, module_i in enumerate(modules):
    module_i.set_thrust_design([modules_setting[1], modules_setting[3]], 0)
    module_i.set_control_function([modules_setting[0], modules_setting[2]])
    final_state = module_i.simulate(tf, low_step=0.01)
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
         np.array(modules[0].dynamics.dynamic_model.historical_theta))
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
