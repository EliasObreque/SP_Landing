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

#%% Data Mars lander
m0 = 1905
Isp = 225
g = -3.7114
ge  = 9.807
T_max = 30000

c_char = Isp * ge

#%% Initial position
r0  = 2000
v0  = 0


#%% Standard dev.
perc = 5    # 0-100%
sdr = r0 * perc / 100
sdv = v0 * perc / 100
sdm = m0 * perc / 100

N_case = 5  # Case number

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
n_thruster = 1
thrust_comp_class = Thruster(dt, max_time=20, max_thrust=T_max)
thrust_comp = []
for i in range(n_thruster):
    thrust_comp.append(copy.deepcopy(thrust_comp_class))
    thrust_comp[i].set_lag_coef(0.2)

time = []
x1   = []
x2   = []
x3   = []
thr  = []
sf   = []


dynamics = Dynamics(dt, Isp, g)


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
    while end_condition is False:
        dynamics.update_control_parameters(x3[i][k], T_max)
        sf[i].append(dynamics.control_sf(x1[i][k], x2[i][k]))
        if sf[i][k] <= 0.0:
            thr[i].append(T_max)
        else:
            thr[i].append(0.0)
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
opt_plot1 = '.b'
opt_plot2 = 'or'
create_plot()
for i in range(N_case):
    set_plot(1, time[i], x1[i], opt_plot1, opt_plot2)
    set_plot(2, time[i], x2[i], opt_plot1, opt_plot2)

plt.show()

print('Finished')





