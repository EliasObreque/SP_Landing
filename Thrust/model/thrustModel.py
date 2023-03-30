"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 28-08-2022
"""
import numpy as np
from thrust.basic_thrust import BasicThruster
from abc import ABC

ge = 9.807
NEUTRAL = 'neutral'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'
gasConstant = 8314.0  # J/ kmol K


class MathModel(BasicThruster, ABC):
    def __init__(self, dt, thruster_properties):
        BasicThruster.__init__(self, dt, thruster_properties)
        # # propellant
        # self.exit_pressure = 1e-12
        # self.amb_pressure = 1e-12
        # selected_propellant = propellant_properties['mixture_name']
        # self.selected_propellant = \
        # [pro_data['data'] for pro_data in propellant_data if pro_data['name'] == selected_propellant][0]
        # self.r_gases = gasConstant / self.selected_propellant['molecular_weight']
        # self.gamma = self.selected_propellant['small_gamma']
        # self.big_gamma = self.get_big_gamma()
        # self.density = self.selected_propellant['density'] * 1e3  # kg/m3
        # self.burn_rate_exponent = self.selected_propellant['pressure_exponent']
        # self.burn_rate_constant = self.selected_propellant['burn_rate_constant'] * 1e-3  # m
        # self.temperature = self.selected_propellant['temperature']  # K
        # self.c_char = (1 / self.big_gamma) * (self.r_gases * self.temperature) ** 0.5  # m/s
        # self.c_f = 0.0
        # self.throat_diameter = thruster_properties['throat_diameter']
        # self.diameter_ext = thruster_properties['case_diameter']
        # self.case_large = thruster_properties['case_large']
        # self.exit_nozzle_diameter = thruster_properties['exit_nozzle_diameter']
        # self.area_th = np.pi * self.throat_diameter ** 2 / 4
        # self.area_exit = np.pi * self.exit_nozzle_diameter ** 2 / 4
        # self.calc_c_f(self.gamma, 1.0)

        # variable for model the tanh function
        self.delay_time_percentage = 0.1
        self.max_value_at_lag_coef = 0.999
        self.percentage_pro_ini = 0.3
        self.percentage_reg_end = 0.3
        self.lag_coef = 0.5
        self.delay_time = 0.2
        self.current_alpha = 0.0
        self.step_width = dt
        self.t_burn = thruster_properties['thrust_profile']['performance']['t_burn']
        self.max_mass_flow = thruster_properties['thrust_profile']['performance']['max_mass_flow']
        self.isp = thruster_properties['thrust_profile']['performance']['isp']
        self.std_noise = thruster_properties['thrust_profile']['performance']['isp_noise_std']
        self.std_bias = thruster_properties['thrust_profile']['performance']['isp_bias_std']
        self.v_exhaust = self.isp * ge
        self.add_bias_isp()
        self.burn_type = thruster_properties['thrust_profile']['performance']['cross_section']
        self.current_burn_time = 0
        self.historical_mag_thrust = [0.0]
        self.t_ig = 0
        self.thr_is_on = False
        self.thr_is_burned = False
        self.current_time = 0
        self.current_mag_thrust_c = 0

        self.dx = (self.lag_coef + self.delay_time) / self.delay_time - 1

        self.percentage_pro_end = self.max_value_at_lag_coef
        self.percentage_reg_ini = self.max_value_at_lag_coef

        dy = (np.arctanh(self.max_value_at_lag_coef * 2 - 1) - np.arctanh(self.delay_time_percentage * 2 - 1))
        self.incline = dy / self.dx
        self.g_displacer_point = 1 - np.arctanh(self.delay_time_percentage * 2 - 1) / self.incline
        self.time_to_rise = (self.g_displacer_point + np.arctanh(self.max_value_at_lag_coef * 2 - 1) / self.incline) * \
                             self.delay_time

        self.time_pro_intersection = (np.arctanh(2 * self.percentage_pro_ini - 1)
                                      / self.incline + self.g_displacer_point) * self.delay_time
        self.time_reg_intersection = self.t_burn + 2 * self.delay_time - \
                                     (np.arctanh(2 * self.percentage_reg_end - 1)
                                      / self.incline + self.g_displacer_point) * self.delay_time
        self.slope_reg = (self.percentage_reg_end - self.percentage_reg_ini) / (
                self.time_reg_intersection - self.delay_time - self.lag_coef)
        self.slope_pro = (self.percentage_pro_end - self.percentage_pro_ini) / (
                self.delay_time + self.t_burn - self.lag_coef - self.time_pro_intersection)
        self.c_pro = self.percentage_pro_ini - self.slope_pro * self.time_pro_intersection
        self.c_reg = self.max_value_at_lag_coef - self.slope_reg * (self.delay_time + self.lag_coef)

    def calc_tanh_model(self, to):
        if to == 'rising':
            return (1 + np.tanh((self.current_burn_time / self.delay_time - self.g_displacer_point) * self.incline)) * 0.5
        elif to == 'decaying':
            return (1 + np.tanh((-self.g_displacer_point - (self.current_burn_time - self.t_burn - 2 * self.delay_time)
                                 / self.delay_time) * self.incline)) * 0.5

    def set_lag_coef(self, val):
        self.lag_coef = val

    def set_alpha(self, value):
        self.current_alpha = value

    def set_t_burn(self, value):
        self.t_burn = value

    def get_current_thrust(self):
        return self.current_mag_thrust_c

    def get_neutral_thrust(self):
        if self.thr_is_on:
            v_exhaust = self.add_noise_isp()
            if self.current_burn_time == 0:
                self.current_mag_thrust_c = 0
                self.current_burn_time += self.step_width
            elif self.current_burn_time <= self.delay_time + self.t_burn/2:
                current_max_thrust = self.max_mass_flow * v_exhaust
                self.current_mag_thrust_c = current_max_thrust * self.calc_tanh_model('rising')
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            elif self.lag_coef + self.delay_time + self.t_burn >= self.current_burn_time:
                current_max_thrust = self.max_mass_flow * v_exhaust
                self.current_mag_thrust_c = current_max_thrust * self.calc_tanh_model('decaying')
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.step_width
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.step_width

    def get_regressive_thrust(self):
        if self.thr_is_on:
            v_exhaust = self.add_noise_isp()
            if self.current_burn_time == 0:
                self.current_mag_thrust_c = 0
                self.current_burn_time += self.step_width
            elif self.current_burn_time <= self.delay_time + self.lag_coef:
                current_max_thrust = self.max_mass_flow * v_exhaust
                self.current_mag_thrust_c = current_max_thrust * self.calc_tanh_model('rising')
                self.current_burn_time += self.step_width
                self.current_time += self.step_width
            elif self.current_burn_time <= self.time_reg_intersection:
                current_max_thrust = self.max_mass_flow * v_exhaust
                self.current_mag_thrust_c = current_max_thrust * (self.slope_reg * self.current_burn_time + self.c_reg)
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            elif self.current_burn_time <= self.delay_time + self.t_burn + self.lag_coef:
                current_max_thrust = self.max_mass_flow * v_exhaust
                self.current_mag_thrust_c = current_max_thrust * self.calc_tanh_model('decaying')
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.step_width
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.step_width

    def get_progressive_thrust(self):
        if self.thr_is_on:
            v_exhaust = self.add_noise_isp()
            if self.current_burn_time == 0:
                self.current_mag_thrust_c = 0
                self.current_burn_time += self.step_width
            elif self.current_burn_time <= self.time_pro_intersection:
                current_max_thrust = self.max_mass_flow * v_exhaust
                self.current_mag_thrust_c = current_max_thrust * self.calc_tanh_model('rising')
                self.current_burn_time += self.step_width
                self.current_time += self.step_width
            elif self.current_burn_time <= self.delay_time + self.t_burn - self.lag_coef:
                current_max_thrust = self.max_mass_flow * v_exhaust
                self.current_mag_thrust_c = current_max_thrust * (self.slope_pro * self.current_burn_time + self.c_pro)
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            elif self.current_burn_time <= self.delay_time + self.t_burn + self.lag_coef:
                current_max_thrust = self.max_mass_flow * v_exhaust
                self.current_mag_thrust_c = current_max_thrust * self.calc_tanh_model('decaying')
                self.current_time += self.step_width
                self.current_burn_time += self.step_width
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.step_width
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.step_width

    def set_t_burn(self, value):
        self.t_burn = value
        self.calc_tanh_model_parameters()

    def calc_tanh_model_parameters(self):
        self.time_pro_intersection = (np.arctanh(2 * self.percentage_pro_ini - 1)
                                      / self.incline + self.g_displacer_point) * self.delay_time
        self.time_reg_intersection = self.t_burn + 2 * self.delay_time - \
                                     (np.arctanh(2 * self.percentage_reg_end - 1)
                                      / self.incline + self.g_displacer_point) * self.delay_time
        self.slope_reg = (self.percentage_reg_end - self.percentage_reg_ini) / (
                self.time_reg_intersection - self.delay_time - self.lag_coef)
        self.slope_pro = (self.percentage_pro_end - self.percentage_pro_ini) / (
                self.delay_time + self.t_burn - self.lag_coef - self.time_pro_intersection)
        self.c_pro = self.percentage_pro_ini - self.slope_pro * self.time_pro_intersection
        self.c_reg = self.max_value_at_lag_coef - self.slope_reg * (self.delay_time + self.lag_coef)

    def get_isp(self):
        return self.isp

    def get_v_exhaust(self):
        return self.v_exhaust

    def add_bias_isp(self):
        if self.std_bias is not None:
            isp = np.random.normal(self.isp, self.std_bias)
            self.v_exhaust = isp * ge

    def add_noise_isp(self):
        if self.std_noise is not None:
            noise_isp = np.random.normal(0, self.std_noise)
            return self.v_exhaust + noise_isp * ge
        else:
            return self.v_exhaust

    def log_value(self):
        self.historical_mag_thrust.append(self.current_mag_thrust_c)

    def reset_variables(self):
        self.t_ig = 0
        self.thr_is_on = False
        self.current_burn_time = 0
        self.current_time = 0
        self.current_mag_thrust_c = 0
        self.thr_is_burned = False

    def propagate_thrust(self):
        if self.burn_type == PROGRESSIVE:
            self.get_progressive_thrust()
        elif self.burn_type == REGRESSIVE:
            self.get_regressive_thrust()
        else:
            self.get_neutral_thrust()
