"""
Created: 7/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .PropellantGrain import PropellantGrain
DEG2RAD = np.pi/180

LINEAR  = 'linear'
TUBULAR = 'tubular'
BATES   = 'bates'
STAR    = 'star'


class Thruster(object):
    def __init__(self, dt, thruster_properties, propellant_properties):
        self.thrust_profile = None
        self.propellant_geometry = propellant_properties['propellant_geometry']
        if thruster_properties['engine_diameter_ext'] is not None:
            throat_diameter = thruster_properties['throat_diameter']
            engine_diameter_ext = thruster_properties['engine_diameter_ext']
            height = thruster_properties['height']
            volume_convergent_zone = (np.pi * height * (engine_diameter_ext * 0.5) ** 2) / 3
            self.volume_case = volume_convergent_zone
        self.step_width = dt
        self.selected_propellant = PropellantGrain(dt, propellant_properties)
        if self.selected_propellant.geometry_grain is not None:
            self.volume_case += self.selected_propellant.selected_geometry.free_volume
        else:
            self.t_burn = thruster_properties['performance']['t_burn']
            self.current_alpha = thruster_properties['performance']['alpha']
        if thruster_properties['load_thrust_profile']:
            self.thrust_profile, self.time_profile = self.load_thrust_profile(thruster_properties['file_name'])
            self.dt_profile = self.time_profile[1] - self.time_profile[0]
        self.current_burn_time = 0
        self.historical_mag_thrust = []
        self.t_ig = 0
        self.thr_is_on = False
        self.thr_is_burned = False
        self.current_time = 0
        self.current_mag_thrust_c = 0
        self.lag_coef = 0.2
        self.current_beta = 0

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

    def propagate_thr(self):
        if self.selected_propellant.geometry_grain is not None:
            """Propagate propellant"""
        elif self.thrust_profile is not None:
            """Propagate loaded profile"""
            self.calc_parametric_thrust()
        else:
            """Propagate an constant thrust"""
            self.get_constant_thrust()
        return

    def get_constant_thrust(self):
        if self.thr_is_on:
            if self.current_burn_time <= self.t_burn:
                self.current_mag_thrust_c = self.current_alpha * self.selected_propellant.c_char
                self.current_burn_time += self.step_width
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.step_width
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.step_width
        return

    def get_current_thrust(self):
        return self.current_mag_thrust_c

    def set_alpha(self, value):
        self.current_alpha = value

    def set_t_burn(self, value):
        self.t_burn = value

    def calc_parametric_thrust(self):
        if self.thr_is_on:
            if self.current_burn_time == 0:
                self.current_mag_thrust_c = self.thrust_profile[0]
                self.current_burn_time += self.step_width
            elif self.current_burn_time <= max(self.time_profile):
                self.current_mag_thrust_c = self.thrust_profile[int(self.current_burn_time / self.dt_profile)]
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

    def calc_linear_thrust(self, com_period_):
        if self.thr_is_on:
            if self.current_burn_time <= self.max_burn_time:
                self.current_mag_thrust_c = self.nominal_thrust
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.step_width
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.step_width

    def set_beta(self, beta, n_engine=0):
        if self.thr_is_burned:
            self.thr_is_on = False
        else:
            if beta == 1 and self.current_beta == 0:
                self.current_beta = beta
                self.t_ig = self.current_time
                self.thr_is_on = True
                # print('thrust ', n_engine, ' ON')
            elif beta == 1 and self.current_beta == 1:
                self.current_beta = beta
            elif self.thr_is_on:
                self.current_beta = 1
            else:
                self.current_beta = 0
                self.thr_is_on = False

    def log_value(self):
        self.historical_mag_thrust.append(self.current_mag_thrust_c)

    @staticmethod
    def load_thrust_profile(file_name):
        dataframe = pd.read_csv("Thrust/StarGrain7.csv")
        return dataframe['Thrust(N)'].values, dataframe['Time(s)'].values


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









