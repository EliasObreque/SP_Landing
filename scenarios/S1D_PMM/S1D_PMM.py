"""
Created by:

@author: Elias Obreque
@Date: 6/5/2021 7:30 PM 
els.obrq@gmail.com

"""

from datetime import datetime
from tools.ext_requirements import mass_req
from dynamics.Dynamics import Dynamics
from thrust.propellant.propellant import propellant_data
from tools.Viewer import *

if os.path.isdir("../logs/") is False:
    os.mkdir("../logs/")

CONSTANT  = 'constant'
TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'

now = datetime.now()
now = now.strftime("%Y-%m-%dT%H-%M-%S")
reference_frame = '1D'
# -----------------------------------------------------------------------------------------------------#
# Data Mars lander (12U (24 kg), 27U (54 kg))

m0 = 24
propellant_name = 'CDT(80)'
selected_propellant = propellant_data[propellant_name]
propellant_geometry = TUBULAR
Isp = selected_propellant['Isp']
den_p = selected_propellant['density']
ge = 9.807
c_char = Isp * ge

# Available space for engine (square)
space_max = 180  # mm
thickness_case_factor = 1.2
aux_dimension = 150  # mm
d_int = 2  # mm

# -----------------------------------------------------------------------------------------------------#
# Center body data
# Moon
g_center_body = -1.62
r_moon = 1738e3
mu = 4.9048695e12

# -----------------------------------------------------------------------------------------------------#
# Initial position for 1D
r0 = 2000
v0 = 0

# Target localization
rd = 0
vd = 0

# -----------------------------------------------------------------------------------------------------#
# Initial requirements for 1D
print('--------------------------------------------------------------------------')
print('1D requirements')
dv_req = np.sqrt(2 * r0 * np.abs(g_center_body))
print('Accumulated velocity[m/s]: ', dv_req)
mp, m1 = mass_req(dv_req, c_char, den_p, m0)

# -----------------------------------------------------------------------------------------------------#
# Simulation time

dt = 0.1
simulation_time = 100
# -----------------------------------------------------------------------------------------------------#
# System Propulsion properties

t_burn_min = 3  # s
t_burn_max = 40  # s
n_thruster = 10
par_force = 1  # Engines working simultaneously

pulse_thruster = int(n_thruster / par_force)

total_alpha_min = - g_center_body * m0 / c_char

# for 1D
max_fuel_mass = 1.1 * mp  # Factor: 1.05

total_alpha_max = max_fuel_mass / t_burn_min
print('Mass flow rate (1D): (min, max) [kg/s]', total_alpha_min, total_alpha_max)


print('Required engines: (min-min, min-max, max-min, max-max) [-]',
      max_fuel_mass / total_alpha_min / t_burn_min,
      max_fuel_mass / total_alpha_min / t_burn_max,
      max_fuel_mass / total_alpha_max / t_burn_min,
      max_fuel_mass / total_alpha_max / t_burn_max)

T_min = total_alpha_min * c_char
T_max = total_alpha_max * c_char
print('Max thrust: (min, max) [N]', T_min, T_max)
print('--------------------------------------------------------------------------')

# -----------------------------------------------------------------------------------------------------#
# Create dynamics object for 1D
dynamics = Dynamics(dt, Isp, g_center_body, mu, r_moon, m0, reference_frame, controller='basic_hamilton')
# -----------------------------------------------------------------------------------------------------#
# Simple example solution with constant one engine for 1D
dynamics.calc_limits_by_single_hamiltonian(t_burn_min, t_burn_max, total_alpha_min, total_alpha_max, plot_data=True)

# Calculate optimal alpha (m_dot) for a given t_burn
max_alt = dynamics.basic_hamilton_calc.x1_hat_max_t
if max_alt < r0:
    t_burn = 0.4 * (t_burn_min + t_burn_max)
    total_alpha_max = 0.95 * m0 / t_burn
else:
    t_burn = 0.3 * (t_burn_min + t_burn_max)
    total_alpha_max = 0.9 * m0 / t_burn

optimal_alpha = dynamics.basic_hamilton_calc.calc_simple_optimal_parameters(r0, total_alpha_min, total_alpha_max,
                                                                            t_burn)

# Define propellant properties to create a Thruster object with the optimal alpha
propellant_properties = {'propellant_name': propellant_name,
                         'n_thrusters': 1,
                         'pulse_thruster': 1,
                         'geometry': None,
                         'propellant_geometry': propellant_geometry,
                         'isp_noise_std': None,
                         'isp_bias_std': None,
                         'isp_dead_time_max': None}

engine_diameter_ext = None
throat_diameter = 1.0  # mm
height = 10.0  # mm
file_name = "thrust/StarGrain7.csv"
thruster_properties = {'throat_diameter': 2,
                       'engine_diameter_ext': engine_diameter_ext,
                       'height': height,
                       'performance': {'alpha': optimal_alpha,
                                       't_burn': t_burn},
                       'load_thrust_profile': False,
                       'file_name': file_name,
                       'delay': None}

dynamics.set_engines_properties(thruster_properties, propellant_properties)

# Initial and final condition
x0 = [r0, v0, m0]
xf = [0.0, 0.0, m1]
time_options = [0, simulation_time, 0.01]

x_states, time_series, thr, _, _, _, _ = dynamics.run_simulation(x0, xf, time_options)
dynamics.basic_hamilton_calc.print_simulation_data(x_states, mp, m0, r0)
dynamics.basic_hamilton_calc.plot_1d_simulation(x_states, time_series, thr)

print("Finished")
