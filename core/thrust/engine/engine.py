"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
from core.thrust.propellant.propellant import Propellant
from scipy.optimize import fsolve
from core.thrust.basic_thrust import BasicThruster
from abc import ABC


class LogChannel:
    def __init__(self, name, valueType, unit):
        if valueType not in (int, float, list, tuple):
            raise TypeError('Value type not in allowed set')
        self.name = name
        self.unit = unit
        self.valueType = valueType
        self.data = []

    def getData(self):
        return np.array(self.data)

    def getPoint(self, i):
        """Returns a specific datapoint by index."""
        return self.data[i]

    def getLast(self):
        """Returns the last datapoint."""
        return self.data[-1]

    def addData(self, data):
        """Adds a new datapoint to the end."""
        self.data.append(data)

    def getAverage(self):
        """Returns the average of the datapoints."""
        if self.valueType in (list, tuple):
            raise NotImplementedError('Average not supported for list types')
        return sum(self.data) / len(self.data)

    def getMax(self):
        """Returns the maximum value of all datapoints. For list datatypes, this operation finds the largest single
        value in any list."""
        if self.valueType in (list, tuple):
            return max([max(l) for l in self.data])
        return max(self.data)


class Engine(BasicThruster, ABC):
    amb_pressure = 1e-12

    def __init__(self, dt, thruster_properties, propellant_properties):
        BasicThruster.__init__(self, dt, thruster_properties, propellant_properties)
        self.count = 0
        self.t_ig = 0.0
        self.thr_is_on = False
        self.current_time = 0.0
        self.current_burn_time = 0.0
        self.historical_mag_thrust = [0.0]
        self.current_mag_thrust_c = 0.0
        self.historical_beta = [self.current_beta]

        # Engine properties
        self.throat_diameter = thruster_properties['throat_diameter']
        self.diameter_ext = thruster_properties['case_diameter']
        self.case_large = thruster_properties['case_large']
        self.divergent_angle = np.deg2rad(thruster_properties['divergent_angle_deg'])
        self.convergent_angle = np.deg2rad(thruster_properties['convergent_angle_deg'])
        self.exit_nozzle_diameter = thruster_properties['exit_nozzle_diameter']
        self.exit_area = np.pi * self.exit_nozzle_diameter ** 2 / 4
        self.throat_area = np.pi * self.throat_diameter ** 2 / 4

        # Engine condition
        d = 0.5 * (self.exit_nozzle_diameter - self.throat_diameter) / np.tan(self.convergent_angle)
        volume_convergent_zone = (np.pi * d * (self.diameter_ext * 0.5) ** 2) / 3
        volume_case = np.pi * ((self.diameter_ext * 0.5) ** 2) * self.case_large
        self.empty_engine_volume = volume_case + volume_convergent_zone
        self.chamber_temperature = 0.0
        self.exit_pressure = self.amb_pressure
        self.chamber_pressure = self.exit_pressure
        self.c_f = 0.0

        self.propellant = Propellant(dt, propellant_properties)
        self.volume_free = self.empty_engine_volume - self.propellant.get_grain_volume()
        self.init_stable_chamber_pressure = self.calc_chamber_pressure(self.propellant.get_burn_area())

        self.channels = {
            'time': LogChannel('Time', float, 's'),
            'kn': LogChannel('Kn', float, ''),
            'pressure': LogChannel('Chamber Pressure', float, 'Pa'),
            'force': LogChannel('Thrust', float, 'N'),
            'mass': LogChannel('Propellant Mass', tuple, 'kg'),
            'volumeLoading': LogChannel('Volume Loading', float, '%'),
            'massFlow': LogChannel('Mass Flow', tuple, 'kg/s'),
            'massFlux': LogChannel('Mass Flux', tuple, 'kg/(m^2*s)'),
            'regression': LogChannel('Regression Depth', tuple, 'm'),
            'web': LogChannel('Web', tuple, 'm'),
            'exitPressure': LogChannel('Nozzle Exit Pressure', float, 'Pa'),
            'dThroat': LogChannel('Change in Throat Diameter', float, 'm')
        }

        # At t = 0, the motor has ignited
        self.channels['time'].addData(0)
        self.channels['kn'].addData(self.calc_kn(0))
        self.channels['pressure'].addData(0)
        self.channels['force'].addData(0)
        self.channels['mass'].addData(self.propellant.get_mass_at_reg(0))
        self.channels['volumeLoading'].addData(100 * (1 - (self.calc_free_volume(0) / self.empty_engine_volume)))
        self.channels['massFlow'].addData(0)
        self.channels['massFlux'].addData(0)
        self.channels['regression'].addData(0)
        self.channels['web'].addData(self.propellant.get_web_left(0))
        self.channels['exitPressure'].addData(0)
        self.channels['dThroat'].addData(0)

        a_b = self.propellant.get_burning_area(0)
        kn = 100
        a_t = a_b / kn
        r_e = np.sqrt(a_t / np.pi)
        self.throat_diameter = r_e * 2
        self.throat_area = np.pi * self.throat_diameter ** 2 / 4
        # self.check_geometric_cond()

    def propagate_thrust(self):
        # print(self.propellant.get_web_left(self.propellant.current_reg_web))
        if self.thr_is_on and not self.thr_was_burned:
            if self.propellant.get_web_left(self.propellant.current_reg_web) > 1e-4:
                p_c = self.channels['pressure'].getLast()
                # calc reg, and burn area
                self.propellant.propagate_grain(p_c)
                # calculate mass properties, flux and flow
                self.propellant.calculate_mass_properties(p_c)
                # new chamber pressure
                self.calc_chamber_pressure(self.propellant.get_burn_area())
                self.calc_exit_pressure(self.propellant.gamma)
                # calc CF
                self.calc_c_f(self.propellant.gamma)
                # calc thrust
                self.calc_thrust(self.propellant.add_noise_isp())
            else:
                self.current_mag_thrust_c = 0
                self.exit_pressure = self.amb_pressure
                self.chamber_pressure = self.exit_pressure
                self.propellant.reset_var()
                self.thr_was_burned = True
            # time
        self.current_time += self.step_width

    def calc_free_volume(self, reg):
        free_vol = self.empty_engine_volume - self.propellant.get_volume_at_reg(reg)
        return free_vol

    def get_chamber_pressure(self):
        return self.chamber_pressure

    def get_current_m_flow(self):
        return self.channels['massFlow'].getLast()

    def calc_thrust(self, noised_isp):
        ratio = self.propellant.isp0 / self.c_f
        new_cf = noised_isp / ratio
        thr = new_cf * self.chamber_pressure * self.throat_area
        self.current_mag_thrust_c = thr

    def calc_chamber_pressure(self, burn_area):
        area_ratio = burn_area / self.throat_area
        pc = (self.propellant.burn_rate_constant * area_ratio * self.propellant.density *
              self.propellant.c_char) ** (1 / (1 - self.propellant.burn_rate_exponent))
        self.chamber_pressure = pc

    def calc_exit_pressure(self, k, p_c=None):
        if p_c is None:
            p_c = self.get_chamber_pressure()
        """Solves for the nozzle's exit pressure, given an input pressure and the gas's specific heat ratio."""
        def expansion(x):
            return np.array(1 / self.calc_expansion() - self.eRatioFromPRatio(k, x / p_c))
        self.exit_pressure = fsolve(expansion,
                                    np.array(0.0))[0]
        return self.exit_pressure

    def calc_expansion(self):
        return (self.exit_area / self.throat_area) ** 2

    def get_isp(self):
        return 0.0

    def get_current_thrust(self):
        return self.current_mag_thrust_c

    @staticmethod
    def eRatioFromPRatio(k, pRatio):
        """Returns the expansion ratio of a nozzle given the pressure ratio it causes."""
        return (((k + 1) / 2) ** (1 / (k - 1))) * (pRatio ** (1 / k)) * (
                    (((k + 1) / (k - 1)) * (1 - (pRatio ** ((k - 1) / k)))) ** 0.5)

    def calc_c_f(self, gamma, p_c=None, exit_press=None):
        if p_c is None:
            p_c = self.chamber_pressure
        if exit_press is None:
            p_e = self.calc_exit_pressure(self.propellant.gamma, p_c)
        else:
            p_e = self.exit_pressure
        a = (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
        gamma_upper = np.sqrt(a * gamma)
        b = 2 * gamma ** 2 / (gamma - 1)
        ratio_p = p_e / p_c
        c = (1 - ratio_p ** ((gamma - 1) / gamma))
        ratio_a = self.exit_area / self.throat_area * (p_e - self.amb_pressure) / p_c
        cf = np.sqrt(b * a * c) + ratio_a
        self.c_f = cf
        return self.c_f

    def get_c_f(self):
        return self.c_f

    def calc_kn(self, burning_surface_area):
        """Returns the motor's Kn when it has each grain has regressed by its value in regDepth, which should be a list
        with the same number of elements as there are grains in the motor."""
        kn_ = burning_surface_area / self.throat_area
        return kn_

    def check_geometric_cond(self):
        init_burn_area = self.propellant.get_port_area(0)
        port_to_throat = self.calc_kn(init_burn_area)
        print("Port-to-Throat: {} - (propellant/Throat) = {}/ {}".format(port_to_throat, init_burn_area,
                                                                         self.throat_area))
        if port_to_throat > 2:
            print("ok")
        else:
            print("Port-to-Throat {} lower than 2".format(port_to_throat))

    def reset_variables(self):
        self.t_ig = 0
        self.thr_is_on = False
        self.current_burn_time = 0
        self.current_time = 0
        self.current_mag_thrust_c = 0
        self.exit_pressure = self.amb_pressure
        self.chamber_pressure = self.exit_pressure
        self.propellant.reset_var(force=True)
        super().reset_variables()

    def set_thrust_on(self, value):
        self.thr_is_on = value

    def log_value(self):
        self.historical_beta.append(self.current_beta)
        self.historical_mag_thrust.append(self.current_mag_thrust_c)
        if self.propellant.isGrain():
            self.channels['time'].addData(self.current_time)
            self.channels['kn'].addData(self.calc_kn(self.propellant.get_burning_area(self.propellant.current_reg_web)))
            self.channels['pressure'].addData(self.chamber_pressure)
            self.channels['force'].addData(self.current_mag_thrust_c)
            self.channels['mass'].addData(self.propellant.mass)
            self.channels['volumeLoading'].addData(100 * (1 - (self.calc_free_volume(self.propellant.current_reg_web) / self.empty_engine_volume)))
            self.channels['massFlow'].addData(self.propellant.mass_flow)
            self.channels['massFlux'].addData(self.propellant.mass_flux)
            self.channels['regression'].addData(self.propellant.current_reg_web)
            self.channels['web'].addData(self.propellant.get_web_left(self.propellant.current_reg_web))
            self.channels['exitPressure'].addData(self.exit_pressure)
            self.channels['dThroat'].addData(0)

    def get_time(self):
        return self.channels['time'].getData()

    def get_mass(self):
        return self.channels['mass'].getLast()

    def set_step_time(self, value):
        self.step_width = value
        self.propellant.dt = value

    def set_bias_isp(self, bias):
        self.propellant.add_bias_isp(bias)
