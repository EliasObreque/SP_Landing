"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""

from Module.Module import Module
import Thrust.ThrustProperties as thr_prop
import Thrust.Propellant.PropellantProperties as prop_prop

n_thrusters = 10
thruster_pos = []
thruster_ang = []
thruster_properties = [thr_prop.default] * n_thrusters
propellant_properties = [prop_prop.propellant_properties] * n_thrusters

reference_frame = '2D'
n_case = 30
mass_0 = 24.0
inertia_0 = 100
current_time = 0
dt = 0.1
tf = 20 * 60
modules = [Module(mass_0, inertia_0, n_thrusters, thruster_pos, thruster_ang, thruster_properties, propellant_properties,
                  reference_frame, dt) for i in range(n_case)]

# Optimal Design of the Control
while current_time <= tf:
    [module_i.update() for module_i in modules]
    [module_i.dynamics.dynamic_model.save_data() for module_i in modules]

    current_time += dt

