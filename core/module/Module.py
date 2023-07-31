"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np

from core.dynamics.Dynamics import Dynamics
from core.thrust.thruster import Thruster


class Module(object):
    thruster_conf = None
    propellant_properties = None
    th = 0.0

    def __init__(self, mass, inertia, init_state, thruster_pos, thruster_ang, thruster_conf,
                 propellant_properties, reference_frame, dt):

        self.thruster_conf = thruster_conf
        self.propellant_properties = propellant_properties
        self.thrusters = [Thruster(dt, thruster_conf[i], propellant_properties[i])() for i in range(len(thruster_conf))]
        self.dynamics = Dynamics(dt, mass, inertia, init_state, reference_frame)
        self.thruster_pos = thruster_pos
        self.thruster_ang = thruster_ang
        self.thrusters_action_wind = [[] for _ in range(len(self.thrusters))]
        self.control_function = self.__default_control
        # self.current_time = 0.0
        # self.dt = dt

    def update(self, control, low_step):
        if low_step is not None:
            self.dynamics.dynamic_model.dt = low_step
            self.dynamics.dynamic_model.h_old = low_step
        for thr_i in self.thrusters:
            thr_i.set_step_time(self.dynamics.dynamic_model.dt)
        [thr_i.set_ignition(control[i]) for i, thr_i in enumerate(self.thrusters)]
        [thr_i.propagate_thrust() for thr_i in self.thrusters]
        [thr_i.log_value() for thr_i in self.thrusters]
        thr = [thr_i.get_current_thrust() for thr_i in self.thrusters]
        m_dot_p = np.sum([thr_i.get_current_m_flow() for thr_i in self.thrusters])
        tau_b = self.calc_torques(thr)
        self.dynamics.dynamic_model.update(np.sum(thr), m_dot_p, np.sum(tau_b), low_step)

    def calc_torques(self, thr_list):
        tau_b = [np.cross(pos_i, thr_i * np.array([0, 1])) for pos_i, thr_i in zip(self.thruster_pos, thr_list)]
        return tau_b

    def save_log(self):
        self.dynamics.dynamic_model.save_data()
        [thr_i.log_value() for thr_i in self.thrusters]

    def get_thrust(self):
        return np.sum([thr_i.current_mag_thrust_c for thr_i in self.thrusters])

    def simulate(self, tf, low_step: float = None, progress: bool = True):
        # save ignition time and stop time
        subk = 0
        k = 0
        control = [0.0] * len(self.thrusters)
        while self.dynamics.dynamic_model.current_time <= tf and self.dynamics.isTouchdown() is False:
            low_step_ = False
            for i, thr in enumerate(self.thrusters):
                control[i] = self.control_function(self.dynamics.get_current_state(), n_e=i)
                if control[i] == 1 and self.thrusters[i].thr_is_burned is False:
                    self.thrusters_action_wind[i].append(subk) if len(self.thrusters_action_wind[i]) == 0 else None
                    low_step_ = sum([True, low_step_])
                else:
                    self.thrusters_action_wind[i].append(subk) if len(self.thrusters_action_wind[i]) == 1 else None
                    low_step_ = sum([False, low_step_])
                if (np.linalg.norm(self.dynamics.dynamic_model.current_pos_i) - 1.738e6) < 2e3:
                    low_step_ = sum([True, low_step_])
            subk += 1
            self.update(control, low_step if low_step_ else None)
            tf_update = self.get_orbit_period()
            self.save_log()
            if tf_update is None:
                break
            if 1.5 * tf_update < tf:
                tf = tf_update
            if tf < self.dynamics.dynamic_model.current_time:
                break
            if k > 29 and progress:
                print('Progress {} % - Thrust: {}'.format(self.dynamics.dynamic_model.current_time / tf * 100,
                                                          self.get_thrust()))
                k = 0
            k += 1
        return self.dynamics.dynamic_model.get_historial()

    def train(self):
        pass

    def evaluate(self):
        pass

    def reset(self):
        [thr.reset_variables() for thr in self.thrusters]
        self.dynamics.dynamic_model.reset()
        self.thrusters_action_wind = [[] for _ in range(len(self.thrusters))]

    def get_orbit_period(self):
        state = self.dynamics.get_current_state()
        mu = self.dynamics.mu
        r = state[0]
        v = state[1]
        energy = 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)
        a = - mu / energy / 2
        if a > 0:
            period = 2 * np.pi * np.sqrt(a ** 3 / mu)
            return period
        else:
            return None

    @staticmethod
    def __default_control(state, n_e=0):
        if state[0][1] < 0.0:
            control = 1
        else:
            control = 1
        return control

    def on_off_control(self, value, n_e=0):
        if n_e == 0:
            value = np.arctan2(value[0][1], value[0][0])
            if value < 0:
                value += 2 * np.pi
            if value >= self.th[n_e]:
                return 1
            else:
                return 0
        elif n_e == 1:
            value = np.arctan2(value[0][1], value[0][0])
            if value < 0:
                value += 2 * np.pi
            if value >= self.th[n_e]:
                return 1
            else:
                return 0
        else:
            value = np.linalg.norm(value[0])
            if value < self.th[n_e]:
                return 1
            else:
                return 0

    def set_control_function(self, control_parameter):
        self.th = control_parameter
        self.control_function = self.on_off_control

    def set_thrust_design(self, thrust_design, orientation):
        for i, value in enumerate(thrust_design):
            self.propellant_properties[i]['geometry']['setting']['ext_diameter'] = value
            self.thruster_conf[i]['case_diameter'] = value
        self.thrusters = [Thruster(self.dynamics.dynamic_model.dt, self.thruster_conf[i], self.propellant_properties[i])() for i in range(len(self.thruster_conf))]

    def get_ignition_state(self, moment):
        values = []
        if moment == 'init':
            for idx_ in self.thrusters_action_wind:
                values.append(self.dynamics.dynamic_model.get_state_idx(idx_[0]))
            return values
        elif moment == 'end':
            for idx_ in self.thrusters_action_wind:
                values.append(self.dynamics.dynamic_model.get_state_idx(idx_[1]))
            return values
        else:
            return None

    def get_mass_used(self):
        return np.sum([thr_i.channels['mass'].getPoint(0) for thr_i in self.thrusters])
