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
TUBULAR = 'tubular'
BATES   = 'bates'
STAR    = 'star'

#%% Data Mars lander
m0      = 1905
Isp     = 225
g       = -3.7114
ge      = 9.807

c_char  = Isp * ge

#%% Initial position
r0  = 5000
v0  = 0

#%% Standard dev.
perc = 0    # 0-100%
sdr = r0 * perc / 100
sdv = v0 * perc / 100
sdm = m0 * perc / 100

N_case = 1  # Case number

#%% Generation of case (Monte Carlo)
rN = MonteCarlo(r0, sdr, N_case).random_value()
vN = MonteCarlo(v0, sdv, N_case).random_value()
mN = MonteCarlo(m0, sdm, N_case).random_value()

#%% Target localitation
rd = 0
vd = 0

#%% Set propellant
dt = 0.01
dead_time = 0.1
t_act_max = 10
t_burn = 0.8 * t_act_max
par_force = 2
n_thruster = int(16 / par_force)
t_burn_total = n_thruster * t_burn

max_fuel_mass = 0.15 * m0
alpha_max = max_fuel_mass/t_burn_total
alpha_min = -g * m0 / c_char + 0.02 * (alpha_max + g * m0 / c_char)
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

dynamics = Dynamics(dt, Isp, g, m0, alpha_min, alpha_max)
dynamics.calc_limits(t_burn_total)
dynamics.show_limits()
optimal_alpha = dynamics.calc_simple_optimal_parameters(r0)
print(optimal_alpha)
# dynamics.calc_optimal_parameters(max_generation=100, n_variables=2, n_individuals=20,
                                   # range_variables=[[alpha_min, alpha_max], [t_burn_total, t_burn_total]])
T_opt = optimal_alpha * c_char
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
    while end_condition is False:
        sf[i].append(dynamics.control_sf(x1[i][k], x2[i][k], x3[i][k], T_opt))
        if sf[i][k] <= 0.0:
            temp_thr = T_opt
        thr[i].append(temp_thr)
        next_state = dynamics.rungeonestep(thr[i][k], x1[i][k], x2[i][k], x3[i][k])
        x1[i].append(next_state[0])
        x2[i].append(next_state[1])
        x3[i].append(next_state[2])

        k += 1
        if x1[i][k] < 0:
            end_condition = True
            x1[i].pop(k)
            x2[i].pop(k)
            x3[i].pop(k)

    time[i] = np.arange(0, len(x1[i])) * dt

#%% plot
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





