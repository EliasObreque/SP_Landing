"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np

from Module.Module import Module
import thrust.ThrustProperties as thr_prop
import thrust.propellant.PropellantProperties as prop_prop

n_thrusters = 10
thruster_pos = []
thruster_ang = []
thruster_properties = [thr_prop.default_thruster] * n_thrusters
propellant_properties = [prop_prop.default_propellant] * n_thrusters

reference_frame = '2D'
n_modules = 10
mass_0 = 24.0
inertia_0 = 100
position = np.array([1, 1])
velocity = np.array([1, 1])
theta = 0
omega = 0
state = [position, velocity, theta, omega]

current_time = 0
dt = 0.1
tf = 20 * 60
modules = [Module(mass_0, inertia_0, state, n_thrusters, thruster_pos, thruster_ang, thruster_properties, propellant_properties,
                  reference_frame, dt) for i in range(n_modules)]

# Optimal Design of the Control
control = []
while current_time <= tf:
    [module_i.update(control[-1]) for module_i in modules]
    [module_i.dynamics.dynamic_model.save_data() for module_i in modules]

    current_time += dt

