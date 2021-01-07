"""
Created: 9/14/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

array_propellant_names = ['JPL_540A', 'ANP-2639AF', 'CDT(80)', 'TRX-H609', 'KNSU']

"""
from matplotlib import pyplot as plt
import numpy as np
from tools.ext_requirements import velocity_req, mass_req
from tools.MonteCarlo import MonteCarlo
from tools.Viewer import create_plot, set_plot
from Dynamics.Dynamics import Dynamics
from Thrust.PropellantGrain import propellant_data

LINEAR = 'linear'
TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'
ONE_D = '1D'
POLAR = 'polar'

reference_frame = ONE_D

# -----------------------------------------------------------------------------------------------------#
# Data Mars lander (12U (24 kg), 27U (54 kg))
m0 = 24
propellant_name = 'CDT(80)'
selected_propellant = propellant_data[propellant_name]
Isp   = selected_propellant['Isp']
den_p = selected_propellant['density']
ge    = 9.807
c_char = Isp * ge

# Available space for engine (square)
space_max = 200  # mm
thickness_case_factor = 1.4
aux_dimension = 100  # mm

# -----------------------------------------------------------------------------------------------------#
# Center body data
# Moon
g_center_body = -1.62
r_moon = 1738e3
mu = 4.9048695e12

# -----------------------------------------------------------------------------------------------------#
# Initial position for 1D
r0 = 2000e3 - r_moon
v0 = 0

# Target localization
rd = 0
vd = 0

# -----------------------------------------------------------------------------------------------------#
# Initial position for Polar coordinate
rrp = 2000e3
rra = 68000e3
ra = 0.5 * (rra + rrp)
# Orbital velocity
vp = np.sqrt(mu * (2 / rrp - 1 / ra))
va = np.sqrt(mu * (2 / rra - 1 / ra))
print('Perilune velocity [m/s]: ', vp)
print('Apolune velocity [m/s]: ', va)
moon_period = 2 * np.pi * np.sqrt(ra ** 3 / mu)

p_r0 = rrp
p_v0 = 0
p_theta0 = 0
p_omega0 = vp / rrp
p_m0 = m0

# Target localization
p_rf = r_moon
p_vf = 0
p_thetaf = 0
p_omegaf = 0
p_mf = m0

# -----------------------------------------------------------------------------------------------------#
# Initial requirements for 1D
print('--------------------------------------------------------------------------')
print('1D requirements')
dv_req = np.sqrt(2 * r0 * np.abs(g_center_body))
print('Accumulated velocity[m/s]: ', dv_req)
mp, m1 = mass_req(dv_req, c_char, den_p, m0)


# Initial requirements for polar
print('\nPolar requirements')
dv_req_p, dv_req_a = velocity_req(vp, va, r_moon, mu, rrp, rra)
p_mp, p_m1 = mass_req(dv_req_p, c_char, den_p, m0)
print('--------------------------------------------------------------------------')

# -----------------------------------------------------------------------------------------------------#
# Simulation time
dt = 0.1
simulation_time = moon_period
# -----------------------------------------------------------------------------------------------------#
# System Propulsion properties
t_burn_min = 4  # s
t_burn_max = 50  # s
n_thruster = 20
par_force  = 2  # Engines working simultaneously

pulse_thruster = int(n_thruster / par_force)

total_alpha_min = - g_center_body * m0 / c_char

# for 1D
total_alpha_max = mp / t_burn_min
print('Mass flow rate (1D): (min, max) [kg/s]', total_alpha_min, total_alpha_max)

# for Polar
total_alpha_max_p = p_mp / t_burn_min
print('Mass flow rate (Polar): (min, max) [kg/s]', total_alpha_min, total_alpha_max_p)

max_fuel_mass = 1.05 * mp  # Factor: 1.05

print('Required engines: (min-min, min-max, max-min, max-max) [-]',
      max_fuel_mass / total_alpha_min / t_burn_min,
      max_fuel_mass / total_alpha_min / t_burn_max,
      max_fuel_mass / total_alpha_max / t_burn_min,
      max_fuel_mass / total_alpha_max / t_burn_max)

print('Required engines: (min-min, min-max, max-min, max-max) [-]',
      max_fuel_mass / total_alpha_min / t_burn_min,
      max_fuel_mass / total_alpha_min / t_burn_max,
      max_fuel_mass / total_alpha_max_p / t_burn_min,
      max_fuel_mass / total_alpha_max_p / t_burn_max)

T_min = total_alpha_min * c_char
T_max = total_alpha_max * c_char
print('Max thrust: (min, max) [N]', T_min, T_max)
print('--------------------------------------------------------------------------')

# -----------------------------------------------------------------------------------------------------#
# Simple one engine example of solution for 1D
dynamics = Dynamics(dt, Isp, g_center_body, mu, r_moon, m0, reference_frame)
dynamics.calc_limits_by_single_hamiltonian(t_burn_min, t_burn_max, total_alpha_min, total_alpha_max)
# Calc. optimal alpha (m_dot)
optimal_alpha = dynamics.basic_hamilton_calc.calc_simple_optimal_parameters(r0, total_alpha_min, total_alpha_max,
                                                                            t_burn_min)
print(optimal_alpha)
x0 = [r0, v0, m0]
xf = [0, 0, m1]
time_options = [0, simulation_time, dt]

x_states, time_series, thr = dynamics.run_simulation(x0, xf, time_options, optimal_alpha, t_burn_min)

plt.figure()
plt.plot(time_series, x_states[:, 0])
plt.show()

# -----------------------------------------------------------------------------------------------------#
# Optimal solution with GA and PMP
n_min_thr, n_max_thr = 10, 10
t_burn_min, t_burn_max = 2, 100

x1_0 = r0
x2_0 = v0
x1_f = 0
x2_f = 0
init_state = [[x1_0, x1_f],
              [x2_0, x2_f],
              m0]


dynamics.calc_optimal_parameters(init_state, max_generation=100, n_individuals=40,
                                 range_variables=[['float', total_alpha_min / n_min_thr, total_alpha_max],
                                                  ['float', t_burn_min, t_burn_max],
                                                  ['int', n_max_thr], ['str', LINEAR],
                                                  ['float_iter', x1_0, x1_f]])
# T_opt = optimal_alpha * c_char
# opt_thruster = Thruster(dt, t_burn_total, nominal_thrust=T_opt, type_propellant=BATES)


time = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
thr = []
psi = []
sf = []

# plot
opt_plot1 = '-b'
opt_plot2 = 'or'
create_plot()

# for i in range(N_case):
#     set_plot(1, time[i], x1[i], opt_plot1, opt_plot2)
#     set_plot(2, time[i], x2[i], opt_plot1, opt_plot2)
#     set_plot(3, time[i], x3[i], opt_plot1, opt_plot2)
#     set_plot(4, time[i], x4[i], opt_plot1, opt_plot2)
#     set_plot(5, time[i], x5[i], opt_plot1, opt_plot2)
#     set_plot(6, time[i], thr[i], opt_plot1, opt_plot2)
#     set_plot(7, time[i], (np.array(x1[i]) - r_moon), opt_plot1, opt_plot2)
#     set_plot(8, np.array(x1[i]) * (np.sin(x3[i])), np.array(x1[i]) * (np.cos(x3[i])), opt_plot1, opt_plot2)
# plt.show()

# thrust_comp_class = Thruster(dt, t_act_max, max_thrust=T_max, type_propellant=STAR)
# thrust_comp = []
# for i in range(n_thruster):
#     thrust_comp.append(copy.deepcopy(thrust_comp_class))
#     thrust_comp[i].set_lag_coef(0.2)

print('Finished')
# Standard dev.
perc = 0  # 0 - 100%
sdr = r0 * perc / 100
sdv = 0 * perc / 100
sdm = 0 * perc / 100

N_case = 1  # Case number

# Generation of case (Monte Carlo)
rN = MonteCarlo(r0, sdr, N_case).random_value()
vN = MonteCarlo(v0, sdv, N_case).random_value()
mN = MonteCarlo(m0, sdm, N_case).random_value()

p_rN = MonteCarlo(p_r0, sdr, N_case).random_value()
p_vN = MonteCarlo(p_v0, sdv, N_case).random_value()
p_thetaN = MonteCarlo(p_theta0, sdv, N_case).random_value()
p_omegaN = MonteCarlo(p_omega0, sdv, N_case).random_value()
p_mN = MonteCarlo(p_m0, sdm, N_case).random_value()