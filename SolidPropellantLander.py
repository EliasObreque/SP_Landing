"""
Created: 9/14/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
from matplotlib import pyplot as plt
import numpy as np
import copy
from Thrust.Thruster import Thruster
from tools.MonteCarlo import MonteCarlo
from tools.Viewer import create_plot, set_plot
from Dynamics.Dynamics import Dynamics
from Thrust.PropellantGrain import PropellantGrain
TUBULAR = 'tubular'
BATES   = 'bates'
STAR    = 'star'

# Data Mars lander (12U (24 kg), 27U (54 kg))
m0      = 24
Isp     = 325
g       = -1.62
ge      = 9.807
den_p   = 1.74

c_char  = Isp * ge

# Initial position 1D
r0  = 5000
v0  = 0

# Initial position Polar
r_moon = 1738e3
mu = 4.9048695e12
rrp = 2000 + r_moon
rra = 68000  + r_moon
ra = 0.5 * (rra + rrp)
# Orbital velocity
vp = np.sqrt(mu * (2 / rrp - 1 / ra))
print('Perilune velocity [m/s]: ', vp)
moon_period = 2 * np.pi * np.sqrt(ra ** 3 / mu)
print('Moon period: ', moon_period / 86400, ' days')
# Falling speed required
vfp = np.sqrt(2 * mu / rrp * (1 - r_moon / rrp))
dv_req_p = vp + vfp


# Mass required
mass_ratio = np.exp(dv_req_p / c_char)
m1 = m0 / mass_ratio
mp = m0 - m1
print('Required mass for propulsion: ', mp, ' [kg]')
print('Required volume for propulsion: ', mp/den_p, ' [cc]')

print('Available mass for payload, structure, and subsystems', m1, ' [kg]')

p_r0 = rrp
p_v0 = 0
p_theta0 = 0
p_omega0 = vp / rrp
p_m0 = m0


# Standard dev.
perc = 0    # 0 - 100%
sdr = r0 * perc / 100
sdv = 0 * perc / 100
sdm = 0 * perc / 100

N_case = 1  # Case number

# Generation of case (Monte Carlo)
rN = MonteCarlo(r0, sdr, N_case).random_value()
vN = MonteCarlo(v0, sdv, N_case).random_value()
mN = MonteCarlo(m0, sdm, N_case).random_value()

p_rN     = MonteCarlo(p_r0, sdr, N_case).random_value()
p_vN     = MonteCarlo(p_v0, sdv, N_case).random_value()
p_thetaN = MonteCarlo(p_theta0, sdv, N_case).random_value()
p_omegaN = MonteCarlo(p_omega0, sdv, N_case).random_value()
p_mN     = MonteCarlo(p_m0, sdm, N_case).random_value()

# Target localitation
rd = 0
vd = 0

# Target localitation
p_rf = r_moon
p_vf = 0
p_thetaf = 0
p_omegaf = 0
p_mf = m0


# Set propellant
dt          = 0.1
dead_time   = 0.1
t_burn_min  = 1
t_burn_max  = 5
par_force   = 2
n_thruster = 30
pulse_thruster  = int(n_thruster / par_force)

max_fuel_mass   = 1.05 * mp
alpha_min       = - g * m0 / c_char * 0.1
alpha_max       = alpha_min * 100.0
print('Mass flow rate: (min, max) [kg/s]', alpha_min, alpha_max)
print('Required engines: (min-min, min-max, max-min, max-max) [-]',
      max_fuel_mass / alpha_min / t_burn_min,
      max_fuel_mass / alpha_min / t_burn_max,
      max_fuel_mass / alpha_max / t_burn_min,
      max_fuel_mass / alpha_max / t_burn_max)

# Available space for engine (square)
space_max = 200  # mm
thickness_case_factor = 1.4
aux_dimension = 100     # mm
# propellant grain
array_propellant_names = ['JPL_540A', 'ANP-2639AF', 'CDT(80)',
                          'TRX-H609', 'KNSU']

propellant_grain_endburn = PropellantGrain(array_propellant_names[2], 2, 30, 100, STAR, 8)
propellant_grain_endburn.simulate_profile(init_pressure, init_r, dt)

T_min   = alpha_min * c_char
T_max   = alpha_max * c_char
print('Max thrust: (min, max) [N]', T_min, T_max)

time = []
x1   = []
x2   = []
x3   = []
x4   = []
x5   = []
thr  = []
psi  = []
sf   = []

polar_system = True
dynamics = Dynamics(dt, Isp, g, m0, alpha_min, alpha_max, 2, 20.0, polar_system=polar_system)
dynamics.calc_limits_const_time(t_burn_total)
dynamics.calc_limits_const_alpha(alpha_max)
dynamics.show_limits()
optimal_alpha = dynamics.calc_simple_optimal_parameters(r0)
print(optimal_alpha)

n_min_thr, n_max_thr = 1, 10
t_burn_min, t_burn_max = 2, 20
x1_0 = r0
x2_0 = v0
x1_f = 0
x2_f = 0
init_state = [[x1_0, x1_f],
              [x2_0, x2_f],
              m0]

sim_time = moon_period

# dynamics.calc_optimal_parameters(init_state, max_generation=100, n_variables=5, n_individuals=20,
#                                  range_variables=[['float', alpha_min, alpha_max], ['float', t_burn_min, t_burn_max],
#                                                   ['int', n_min_thr, n_max_thr], ['str', TUBULAR, BATES, STAR],
#                                                   ['float_iter', x1_0, x1_f]])
T_opt = optimal_alpha * c_char
opt_thruster = Thruster(dt, t_burn_total, nominal_thrust=T_opt, type_propellant=BATES)

for i in range(N_case):
    k = 0
    x1.append([])
    x2.append([])
    x3.append([])
    x4.append([])
    x5.append([])
    sf.append([])
    thr.append([])
    psi.append([])
    time.append([])

    if polar_system:
        x1[i].append(p_rN[i])
        x2[i].append(p_vN[i])
        x3[i].append(p_thetaN[i])
        x4[i].append(p_omegaN[i])
        x5[i].append(p_mN[i])
    else:
        x1[i].append(rN[i])
        x2[i].append(vN[i])
        x3[i].append(mN[i])
        x4[i].append(mN[i])
        x5[i].append(mN[i])

    end_condition = False
    temp_thr = 0
    temp_t_burn = 0
    temp_dt = 0
    while end_condition is False:
        sf[i].append(dynamics.control_sf(x1[i][k], x2[i][k], x3[i][k], T_opt))
        if sf[i][k] <= 0.0:
            temp_thr = T_opt
            opt_thruster.set_beta(1.0)
            temp_dt = dt
        if temp_t_burn >= t_burn_total:
            temp_thr = 0.0
            opt_thruster.set_beta(0.0)

        # opt_thruster.calc_thrust_mag(100)
        # opt_thruster.log_value()
        # print(temp_thr, opt_thruster.current_mag_thrust_c)
        # temp_thr = opt_thruster.current_mag_thrust_c
        temp_t_burn += temp_dt
        thr[i].append(0)
        psi[i].append(0)
        state = [x1[i][k], x2[i][k], x3[i][k], x4[i][k], x5[i][k]]
        next_state = dynamics.rungeonestep(thr[i][k], state, psi[i][k])
        x1[i].append(next_state[0])
        x2[i].append(next_state[1])
        x3[i].append(next_state[2])
        x4[i].append(next_state[3])
        x5[i].append(next_state[4])

        k += 1
        if sim_time < k * dt or (x1[i][k] < 0 and thr[i][k-1] == 0.0):
            end_condition = True
            x1[i].pop(k)
            x2[i].pop(k)
            x3[i].pop(k)
            x4[i].pop(k)
            x5[i].pop(k)

    time[i] = np.arange(0, len(x1[i])) * dt

# plot
opt_plot1 = '-b'
opt_plot2 = 'or'
create_plot()

for i in range(N_case):
    set_plot(1, time[i], x1[i], opt_plot1, opt_plot2)
    set_plot(2, time[i], x2[i], opt_plot1, opt_plot2)
    set_plot(3, time[i], x3[i], opt_plot1, opt_plot2)
    set_plot(4, time[i], x4[i], opt_plot1, opt_plot2)
    set_plot(5, time[i], x5[i], opt_plot1, opt_plot2)
    set_plot(6, time[i], thr[i], opt_plot1, opt_plot2)
    set_plot(7, time[i], (np.array(x1[i]) - r_moon), opt_plot1, opt_plot2)
    set_plot(8, np.array(x1[i]) * (np.sin(x3[i])), np.array(x1[i]) * (np.cos(x3[i])), opt_plot1, opt_plot2)
plt.show()

# thrust_comp_class = Thruster(dt, t_act_max, max_thrust=T_max, type_propellant=STAR)
# thrust_comp = []
# for i in range(n_thruster):
#     thrust_comp.append(copy.deepcopy(thrust_comp_class))
#     thrust_comp[i].set_lag_coef(0.2)

print('Finished')





