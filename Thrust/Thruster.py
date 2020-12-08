"""
Created: 7/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DEG2RAD = np.pi/180

TUBULAR = 'tubular'
BATES   = 'bates'
STAR    = 'star'


class Thruster(object):
    def __init__(self, dt, max_burn_time, nominal_thrust, type_propellant=TUBULAR):
        self.typye_propellant = type_propellant
        self.step_width = dt
        self.max_burn_time = max_burn_time
        self.nominal_thrust = nominal_thrust
        self.parametric_profile = self.load_thrust_profile()
        self.set_nominal_thrust()
        self.current_burn_time = 0
        self.historical_mag_thrust = []
        self.t_ig = 0
        self.thr_is_on = False
        self.thr_is_burned = False
        self.current_time = 0
        self.current_mag_thrust_c = 0
        self.lag_coef = 0.2
        self.current_beta = 0

    def set_nominal_thrust(self):
        med_thrust = np.max(self.parametric_profile)
        factor = self.nominal_thrust / med_thrust
        self.parametric_profile *= factor

    def set_lag_coef(self, val):
        self.lag_coef = val

    def load_thrust_profile(self):
        if self.typye_propellant == STAR:
            dataframe = pd.read_csv("Thrust/StarGrain7.csv")
            self.dt_profile = self.max_burn_time/(len(dataframe['Thrust(N)']) - 1)
            return dataframe['Thrust(N)'].values / max(dataframe['Thrust(N)'].values)
        elif self.typye_propellant == BATES:
            dataframe = pd.read_csv("Thrust/BATES.csv")
            self.dt_profile = self.max_burn_time/(len(dataframe['Thrust(N)']) - 1)
            return dataframe['Thrust(N)'].values / max(dataframe['Thrust(N)'].values)
        else:
            return 1.0

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
        if self.typye_propellant == TUBULAR:
            self.calc_tubular_thrust(com_period_)
        else:
            self.calc_parametric_thrust(com_period_)
        return

    def calc_parametric_thrust(self, com_period_):
        if self.thr_is_on:
            if self.current_burn_time == 0:
                self.current_mag_thrust_c = self.parametric_profile[0]
                self.current_burn_time += self.step_width
            elif self.current_burn_time <= self.max_burn_time:
                self.current_mag_thrust_c = self.parametric_profile[int(self.current_burn_time / self.dt_profile)]
                self.current_burn_time += self.step_width
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.step_width
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.step_width

    def calc_tubular_thrust(self, com_period_):
        if self.thr_is_on:
            if self.current_burn_time <= self.max_burn_time/2:
                ite = 0
                while ite < com_period_ / self.step_width:

                    self.current_mag_thrust_c = self.nominal_thrust * (1 - np.exp(- self.current_burn_time / self.lag_coef))
                    ite += 1
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            elif self.max_burn_time >= self.current_burn_time > self.max_burn_time/2:
                ite = 0
                while ite < com_period_ / self.step_width:
                    self.current_mag_thrust_c = self.nominal_thrust * (1 - np.exp((self.current_burn_time -
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
            elif self.thr_is_on:
                self.current_beta = 1
            else:
                self.current_beta = 0
                self.thr_is_on = False

    def set_max_thrust(self, max_thrust):
        self.max_thrust = max_thrust

    def log_value(self):
        self.historical_mag_thrust.append(self.current_mag_thrust_c)

#%%


if __name__ == '__main__':
    import random

    dt = 0.01
    max_burn_time = 10
    max_thrust = 1
    n_thruster = 5
    betas = []
    time_betas = []
    current_time = 0
    max_time = 30
    comp_thrust = []
    chossing = []
    len_vect = int((max_time/dt)) + 1
    imp_thrust = np.ones(3)

    for i in range(n_thruster):
        comp_thrust.append(Thruster(dt, max_burn_time, max_thrust, type_propellant=BATES))
        comp_thrust[i].set_lag_coef(0.15)
        betas.append(np.zeros(len_vect))
        chossing.append(random.randint(0, int(((max_time - max_burn_time)/dt)) + 1))
        betas[i][chossing[i]: chossing[i] + int(max_burn_time/dt)] = 1

    time_array = []
    k = 1
    while current_time <= max_time:
        time_array.append(current_time)
        for i in range(n_thruster):
            comp_thrust[i].set_beta(betas[i][k-1])
            comp_thrust[i].calc_thrust_mag(100)
            comp_thrust[i].log_value()
        current_time = round(current_time, 3) + dt
        k += 1
    #%%

    fig = plt.figure(figsize=(7, 4))
    axes = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    total_thrust = np.zeros(len(time_array))
    for i in range(n_thruster):
        plt.step(time_array, comp_thrust[i].historical_mag_thrust, '--', label='Engine: '+str(i+1), lw=1.0)
        total_thrust += np.array(comp_thrust[i].historical_mag_thrust)
    plt.step(time_array, total_thrust, 'k', label='Total', lw=1.0)
    plt.legend(loc="center right", borderaxespad=-9.5)
    plt.ylabel('Thrust [-]')
    plt.xlabel('Time [s]')
    plt.grid()

#%%
    fig_beta = plt.figure(figsize=(7, 2))
    axes = fig_beta.add_axes([0.1, 0.3, 0.7, 0.6])
    plt.step(np.arange(0, len(betas[0]))*dt, betas[0], label=r'$\beta_k$: '+str(1), lw=1.0)
    plt.legend(loc="center right", borderaxespad=-9.5)
    plt.ylabel(r'$\beta_k [-]$')
    plt.xlabel('Time [s]')
    plt.grid()
    axes.annotate('Ignition time '+r'$t_{ig}$', xy=(time_array[chossing[0]], 0.), xytext=(-1.5, 0.6),
                  arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=90,angleB=0"))
    axes.annotate('Burn time ' + r'$t_{b}$', xy=(time_array[chossing[0]], 0.5),
                  xytext=(time_array[chossing[0]] + max_burn_time, 0.5),
                  arrowprops=dict(arrowstyle="<->", connectionstyle="angle3,angleA=90,angleB=0"))

    fig_0 = plt.figure(figsize=(7, 2))
    axes = fig_0.add_axes([0.1, 0.3, 0.7, 0.6])
    plt.step(time_array, comp_thrust[0].historical_mag_thrust, label='Engine: '+str(1), lw=1.0)
    plt.plot(time_array[chossing[0]], 0., 'bo')
    plt.legend(loc="center right", borderaxespad=-9.5)
    plt.ylabel('Thrust [-]')
    plt.xlabel('Time [s]')
    plt.grid()
    axes.annotate('Ignition time '+r'$t_{ig}$', xy=(time_array[chossing[0]], 0.), xytext=(-1.5, 0.6),
                  arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=90,angleB=0"))

    plt.show()









