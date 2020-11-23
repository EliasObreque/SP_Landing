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


# Data Mars lander
m0      = 1905
Isp     = 225
g       = -3.7114
ge      = 9.807

c_char  = Isp * ge

# Initial position
r0  = 5000
v0  = 0

# Standard dev.
perc = 5    # 0-100%
sdr = r0 * perc / 100
sdv = 0 * perc / 100
sdm = 0 * perc / 100

N_case = 10  # Case number

# Generation of case (Monte Carlo)
rN = MonteCarlo(r0, sdr, N_case).random_value()
vN = MonteCarlo(v0, sdv, N_case).random_value()
mN = MonteCarlo(m0, sdm, N_case).random_value()

# Target localitation
rd = 0
vd = 0

# Set propellant
dt = 0.01
dead_time = 0.1
t_act_max = 10
t_burn = 0.8 * t_act_max
par_force = 2
n_thruster = int(16 / par_force)
t_burn_total = n_thruster * t_burn

max_fuel_mass = 0.15 * m0
alpha_max = max_fuel_mass/t_burn_total
alpha_min = - g * m0 / c_char + 0.02 * (alpha_max + g * m0 / c_char)
print(alpha_min, alpha_max)

T_min   = alpha_min * c_char
T_max   = alpha_max * c_char
print(T_min, T_max)

time = []
x1   = []
x2   = []
x3   = []
thr  = []
sf   = []

dynamics = Dynamics(dt, Isp, g, m0, alpha_min, alpha_max, 20.0, 80.0)
dynamics.calc_limits_const_time(t_burn_total)
dynamics.calc_limits_const_alpha(alpha_max)
dynamics.show_limits()
optimal_alpha = dynamics.calc_simple_optimal_parameters(r0)
print(optimal_alpha)

TUBULAR = 'tubular'
BATES   = 'bates'
STAR    = 'star'

n_min_thr, n_max_thr = 2, 20
t_burn_min, t_burn_max = 2, 20
x1_0 = r0
x2_0 = v0
x1_f = 0
x2_f = 0
init_state = [[x1_0, x1_f],
              [x2_0, x2_f],
              m0]

dynamics.calc_optimal_parameters(init_state, max_generation=100, n_variables=4, n_individuals=20,
                                 range_variables=[['float', alpha_min, alpha_max], ['float', t_burn_min, t_burn_max],
                                                  ['int', n_min_thr, n_max_thr], ['str', TUBULAR, BATES, STAR]])
T_opt = optimal_alpha * c_char
opt_thruster = Thruster(dt, t_burn_total, nominal_thrust=T_opt, type_propellant=BATES)

for i in range(N_case):
    k = 0
    x1.append([])
    x2.append([])
    x3.append([])
    sf.append([])
    thr.append([])
    time.append([])

    x1[i].append(rN[i])
    x2[i].append(vN[i])
    x3[i].append(mN[i])
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
        thr[i].append(temp_thr)
        next_state = dynamics.rungeonestep(thr[i][k], x1[i][k], x2[i][k], x3[i][k])
        x1[i].append(next_state[0])
        x2[i].append(next_state[1])
        x3[i].append(next_state[2])

        k += 1
        if x1[i][k] < 0 and thr[i][k-1] == 0.0:
            end_condition = True
            x1[i].pop(k)
            x2[i].pop(k)
            x3[i].pop(k)

    time[i] = np.arange(0, len(x1[i])) * dt

# plot
opt_plot1 = '-b'
opt_plot2 = 'or'
create_plot()
for i in range(N_case):
    set_plot(1, time[i], x1[i], opt_plot1, opt_plot2)
    set_plot(2, time[i], x2[i], opt_plot1, opt_plot2)
    set_plot(3, x1[i], x2[i], opt_plot1, opt_plot2)
    set_plot(4, time[i], x3[i], opt_plot1, opt_plot2)
    set_plot(5, time[i], thr[i], opt_plot1, opt_plot2)


plt.show()

# thrust_comp_class = Thruster(dt, t_act_max, max_thrust=T_max, type_propellant=STAR)
# thrust_comp = []
# for i in range(n_thruster):
#     thrust_comp.append(copy.deepcopy(thrust_comp_class))
#     thrust_comp[i].set_lag_coef(0.2)

print('Finished')





