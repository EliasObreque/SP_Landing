"""
Created: 7/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
import numpy as np
import pandas as pd
from thrust.engine.engine import Engine
from thrust.model.thrustModel import MathModel
from thrust.thrustProperties import MODEL, GRAIN, FILE
from scipy.optimize import fsolve
DEG2RAD = np.pi/180


class Thruster(MathModel, Engine):
    def __init__(self, dt, thruster_properties, propellant_properties):
        self.step_width = dt
        self.thrust_by_file = None
        self.thrust_profile_type = thruster_properties['thrust_profile']['type']
        self.max_dead_time = thruster_properties['max_ignition_dead_time']
        self.ignition_dead_time = np.random.uniform(0, self.max_dead_time)

        if self.thrust_profile_type == MODEL:
            super(Thruster, self).__init__(dt, thruster_properties,
                                           propellant_properties)
            Engine.__init__(self, dt, thruster_properties, propellant_properties)
            # self.model = MathModel(dt, thruster_properties['thrust_profile']['performance'])

        elif self.thrust_profile_type == FILE:
            self.thrust_by_file, self.time_profile = self.load_thrust_profile(thruster_properties['thrust_profile']['file_name'])
            self.isp_by_file = thruster_properties['thrust_profile']['isp']
            self.dt_profile = self.time_profile[1] - self.time_profile[0]

        elif self.thrust_profile_type == GRAIN:
            Engine.__init__(self, dt, thruster_properties, propellant_properties)
        else:
            print('Error selecting thrust profile simulation')

        self.historical_mag_thrust = []
        self.current_mag_thrust_c = 0
        self.current_beta = 0
        self.current_dead_time = 0

    def set_lag_coef(self, val):
        self.lag_coef = val

    def reset_variables(self):
        super().reset_variables()
        self.current_beta = 0
        self.reset_dead_time()

    def reset_dead_time(self):
        self.ignition_dead_time = np.random.uniform(0, self.max_dead_time)
        self.current_dead_time = 0.0
        return

    def propagate_thrust(self):
        if self.thrust_profile_type == GRAIN:
            """Propagate propellant by grain set"""
            self.propagate_engine()

        elif self.thrust_profile_type == FILE:
            """Propagate loaded profile by file"""
            self.get_load_thrust()

        elif self.thrust_profile_type == MODEL:
            """Propagate thrust by model"""
            self.propagate_model()
        else:
            print('Error propagating thrust profile simulation')
        return

    def get_current_thrust(self):
        return self.current_mag_thrust_c

    def get_current_m_flow(self):
        return self.get_current_thrust() / self.propellant.get_isp()

    def set_alpha(self, value):
        self.current_alpha = value

    def get_isp(self):
        if self.thrust_profile_type == MODEL:
            isp = self.isp

        elif self.thrust_profile_type == FILE:
            isp = self.isp_by_file

        elif self.thrust_profile_type == GRAIN:
            isp = self.propellant.get_isp()
        else:
            isp = 150
        return isp

    def set_t_burn(self, value):
        self.t_burn = value

    def get_load_thrust(self):
        if self.thr_is_on:
            if self.current_burn_time == 0:
                self.current_mag_thrust_c = self.thrust_by_file[0]
                self.current_burn_time += self.step_width
            elif self.current_burn_time <= max(self.time_profile):
                self.current_mag_thrust_c = self.thrust_by_file[int(self.current_burn_time / self.dt_profile)]
                self.current_burn_time += self.step_width
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.step_width
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.step_width

    def set_ignition(self, beta):
        if self.thr_is_burned:
            self.set_thrust_on(False)
        else:
            if beta == 1 and self.current_beta == 0:
                if self.current_dead_time >= self.ignition_dead_time:
                    self.current_beta = beta
                    self.t_ig = self.current_time
                    self.set_thrust_on(True)
                else:
                    self.update_dead_time()
            elif beta == 1 and self.current_beta == 1:
                self.current_beta = beta
            elif self.thr_is_on:
                self.current_beta = 1
            else:
                self.current_beta = 0
                self.set_thrust_on(False)

    def update_dead_time(self):
        self.current_dead_time += self.step_width
        return

    def log_value(self):
        self.historical_mag_thrust.append(self.current_mag_thrust_c)

    @staticmethod
    def load_thrust_profile(file_name):
        dataframe = pd.read_csv("thrust/" + file_name)
        return dataframe['thrust(N)'].values, dataframe['Time(s)'].values

    def calc_area_by_mass_flow(self, m_dot):
        p_c_ = fsolve(lambda x: m_dot * self.propellant.isp0 * ge - self.calc_c_f(self.propellant.gamma, x, exit_press=None) * x * self.area_th, 10000.0, full_output=1)
        p_c = p_c_[0][0]
        area_p = self.area_th * p_c ** (1 - self.propellant.burn_rate_exponent) /\
                 self.propellant.burn_rate_constant / self.propellant.density / self.propellant.c_char
        r = np.sqrt(area_p / np.pi)
        print("Pressure [kPa]:", p_c * 1e-3)
        print("Area: ", area_p, "Diameter:", r * 2)
        d = 1.5 * p_c * r / 110e6
        print(d)
        if d < 1e-3:
            d = 1e-3
        print("Thickness [mm]:", d * 1e3)
        print("Mass Engine: ", 2 * np.pi * d * r * 0.22 * 2700)
        return r * 1e3


if __name__ == '__main__':
    from thrust.propellant.propellantProperties import *
    from thrustProperties import default_thruster
    from tools.Viewer import plot_thrust, show_plot
    from thrust.propellant.source.propellant_data import propellant_data

    NEUTRAL = 'neutral'
    PROGRESSIVE = 'progressive'
    REGRESSIVE = 'regressive'
    m0 = 24
    ge = 9.807
    dt = 0.01

    file_name = "thrust/StarGrain7.csv"

    mixture_name = 'TRX-H609'
    propellant_data_ = [pro_data for pro_data in propellant_data if pro_data['name'] == mixture_name][0]
    Isp = propellant_data_['data']['Isp']
    propellant_properties_ = default_propellant
    propellant_properties_['mixture_name'] = mixture_name
    propellant_properties_['geometry']['type'] = BATES
    if propellant_properties_['geometry']['type'] is not None:
        propellant_properties_['geometry']['setting'] = bates_geom

    percentage_variation_n = 3
    upper_isp_noise = Isp * (1.0 + percentage_variation_n / 100.0)
    propellant_properties_['isp_noise_std'] = (upper_isp_noise - Isp) / 3

    percentage_variation_b = 10
    upper_isp_bias = Isp * (1.0 + percentage_variation_b / 100.0)
    propellant_properties_['isp_bias_std'] = (upper_isp_bias - Isp) / 3

    thruster_properties_ = default_thruster
    thruster_properties_['thrust_profile']['type'] = MODEL
    thruster_properties_['max_ignition_dead_time'] = 0.5

    if thruster_properties_['thrust_profile']['type'] == MODEL:
        thruster_properties_['thrust_profile']['performance']['cross_section'] = REGRESSIVE
        thruster_properties_['thrust_profile']['performance']['isp'] = Isp
        thruster_properties_['thrust_profile']['performance']['isp_noise_std'] = (upper_isp_noise - Isp) / 3
        thruster_properties_['thrust_profile']['performance']['isp_bias_std'] = (upper_isp_bias - Isp) / 3

    ctrl_a = [1.0]
    ctrl_b = [6.91036]
    max_mass_flow = 1 / Isp / ge
    t_burn = 5
    json_list = {'1': {'Best_individual': [max_mass_flow, t_burn, ctrl_a, ctrl_b]}}
    thruster_properties_['thrust_profile']['performance']['t_burn'] = t_burn
    thruster_properties_['thrust_profile']['performance']['max_mass_flow'] = max_mass_flow

    n_thruster = 1
    comp_thrust = []
    for i in range(n_thruster):
        comp_thrust.append(Thruster(dt, thruster_properties_, propellant_properties_))

    # IDEAL CASE
    propellant_properties_['isp_noise_std'] = None
    propellant_properties_['isp_bias_std'] = None

    if thruster_properties_['thrust_profile']['type'] == MODEL:
        thruster_properties_['thrust_profile']['performance']['isp_noise_std'] = 0.0
        thruster_properties_['thrust_profile']['performance']['isp_bias_std'] = 0.0

    thruster_properties_['max_ignition_dead_time'] = 0.0

    comp_thrust_free = Thruster(dt, thruster_properties_, propellant_properties_)

    time_array = []
    k = 1
    current_time = 0
    beta = 0
    while current_time <= 2 * t_burn + 5:
        time_array.append(current_time)
        thr = 0
        if current_time >= 2:
            beta = 1
        for i in range(n_thruster):
            comp_thrust[i].set_ignition(beta)
            comp_thrust[i].propagate_thrust()
            comp_thrust[i].log_value()

        comp_thrust_free.set_ignition(beta)
        comp_thrust_free.propagate_thrust()
        comp_thrust_free.log_value()

        current_time += dt

    total_thrust = 0
    torque = 0
    for hist in comp_thrust:
        total_thrust += np.array(hist.historical_mag_thrust)

    print("radius mm: ", comp_thrust[0].calc_area_by_mass_flow(10.16e-3))

    plot_thrust(time_array, total_thrust, comp_thrust_free.historical_mag_thrust,
                ['Model thrust [N]', 'Ideal thrust [N]'], dead=0.0)

    show_plot()
