"""
Created: 7/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
#%%
import numpy as np

DEG2RAD = np.pi/180


class Thruster(object):
    def __init__(self, dt, max_time, max_thrust):
        self.step_width = dt
        self.max_thrust_time = max_time
        self.current_burn_time = 0
        self.max_thrust = max_thrust
        self.historical_mag_thrust = []
        self.current_thrust_i = np.zeros(3)
        self.t_ig = 0
        self.thr_is_on = False
        self.thr_is_burned = False
        self.unit_vector_control_i = np.array([0, 0, 1])
        self.current_time = 0
        self.current_mag_thrust_c = 0
        self.lag_coef = 1

    def set_lag_coef(self, val):
        self.lag_coef = val

    def reset_variables(self):
        self.t_ig = 0
        self.thr_is_on = False
        self.current_beta = 0
        self.current_burn_time = 0
        self.current_time = 0
        self.current_mag_thrust_c = 0
        self.thr_is_burned = False

    def set_mag_thrust(self, max_thrust):
        self.max_thrust = max_thrust

    def calc_thrust_mag(self, com_period_):
        com_period_ /= 1000
        if self.thr_is_on:
            if self.current_burn_time <= self.max_burn_time/2:
                ite = 0
                while ite < com_period_ / self.step_width:
                    self.current_mag_thrust_c = self.max_thrust * (1 - np.exp(- self.current_burn_time / self.lag_coef))
                    ite += 1
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            elif self.max_burn_time >= self.current_burn_time > self.max_burn_time/2:
                ite = 0
                while ite < com_period_ / self.step_width:
                    self.current_mag_thrust_c = self.max_thrust * (1 - np.exp((self.current_burn_time -
                                                                               self.max_burn_time) / self.lag_coef))
                    ite += 1
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.step_width
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.step_width

    def set_beta(self, beta):
        if self.thr_is_burned:
            self.thr_is_on = False
        else:
            if beta == 1 and self.current_beta == 0:
                self.current_beta = beta
                self.t_ig = self.current_time
                self.thr_is_on = True
            elif beta == 1 and self.current_beta == 1:
                self.current_beta = beta
            else:
                self.current_beta = 0
                self.thr_is_on = False

    def set_max_thrust(self, max_thrust):
        self.max_thrust = max_thrust

    def log_value(self):
        self.historical_mag_thrust.append(self.current_mag_thrust_c)

#%%


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random

    thruster_pos = np.zeros(3)
    thruster_pos[0] = 0.09
    thruster_pos[1] = 0.13
    thruster_pos[2] = 0.0
    thruster_dir = np.zeros(3)
    thruster_dir[0] = 0
    thruster_dir[1] = 0
    thruster_dir[2] = 1
    step_prop = 0.01
    burn_time = 7.5
    data = {'prop_step': step_prop,
            'comp_type': 'SOLID',
            'thruster_pos': thruster_pos,
            'thruster_dir': thruster_dir,
            'error_pos': 1.0,
            'error_dir': np.deg2rad(3),
            'rotation_z': 0,
            'max_thrust': 1.0,
            'burn_time': burn_time}
    n_thruster = 1
    betas = []
    time_betas = []
    current_time = 0
    max_time = 30
    comp_thrust = []
    chossing = []
    len_vect = int((max_time/step_prop)) + 1
    imp_thrust = np.ones(3)
    for i in range(n_thruster):
        comp_thrust.append(Thruster(6, data))
        comp_thrust[i].set_lag_coef(0.15)
        betas.append(np.zeros(len_vect))
        chossing.append(random.randint(0, int(((max_time - burn_time)/step_prop)) + 1))
        betas[i][chossing[i]: chossing[i] + int(burn_time/step_prop)] = 1

    time_array = []
    k = 1
    while current_time <= max_time:
        time_array.append(current_time)
        for i in range(n_thruster):
            comp_thrust[i].set_beta(betas[i][k-1])
            comp_thrust[i].log_value()
        current_time = round(current_time, 3) + step_prop
        k += 1
    #%%
    fig = plt.figure(figsize=(7, 4))
    axes = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    total_thrust = np.zeros(len(time_array))
    for i in range(n_thruster):
        plt.plot(time_array, comp_thrust[i].historical_mag_thrust, '--', label='Engine: '+str(i+1), lw=1.0)
        total_thrust += np.array(comp_thrust[i].historical_mag_thrust)
    plt.plot(time_array, total_thrust, 'k', label='Total', lw=1.0)
    plt.legend(loc="center right", borderaxespad=-9.5)
    plt.ylabel('Thrust [-]')
    plt.xlabel('Time [s]')
    plt.grid()

#%%
    fig_beta = plt.figure(figsize=(7, 2))
    axes = fig_beta.add_axes([0.1, 0.3, 0.7, 0.6])
    plt.step(np.arange(0, len(betas[0]))*step_prop, betas[0], label=r'$\beta_k$: '+str(1), lw=1.0)
    plt.legend(loc="center right", borderaxespad=-9.5)
    plt.ylabel(r'$\beta_k [-]$')
    plt.xlabel('Time [s]')
    plt.grid()
    axes.annotate('Ignition time '+r'$t_{ig}$', xy=(time_array[chossing[0]], 0.), xytext=(-1.5, 0.6),
                  arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=90,angleB=0"))
    axes.annotate('Burn time ' + r'$t_{b}$', xy=(time_array[chossing[0]], 0.5),
                  xytext=(time_array[chossing[0]] + burn_time, 0.5),
                  arrowprops=dict(arrowstyle="<->", connectionstyle="angle3,angleA=90,angleB=0"))

    fig_0 = plt.figure(figsize=(7, 2))
    axes = fig_0.add_axes([0.1, 0.3, 0.7, 0.6])
    plt.plot(time_array, comp_thrust[0].historical_mag_thrust, label='Engine: '+str(1), lw=1.0)
    plt.plot(time_array[chossing[0]], 0., 'bo')
    plt.legend(loc="center right", borderaxespad=-9.5)
    plt.ylabel('Thrust [-]')
    plt.xlabel('Time [s]')
    plt.grid()
    axes.annotate('Ignition time '+r'$t_{ig}$', xy=(time_array[chossing[0]], 0.), xytext=(-1.5, 0.6),
                  arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=90,angleB=0"))

    plt.show()









