"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
from core.dynamics.Dynamics import Dynamics
from core.thrust.thruster import Thruster

rm = 1.738e6


class Module(object):
    thruster_conf = None
    propellant_properties = None
    th = 0.0

    def __init__(self, mass, inertia, init_state, thruster_pos, thruster_ang, thruster_conf,
                 propellant_properties, reference_frame, dt, training=False):
        self.training = training
        self.thruster_conf = thruster_conf
        self.propellant_properties = propellant_properties
        self.thrusters = [Thruster(dt, thruster_conf[i], propellant_properties[i])() for i in range(len(thruster_conf))]
        self.dynamics = Dynamics(dt, mass, inertia, init_state, reference_frame)
        self.thruster_pos = thruster_pos
        self.thruster_ang = thruster_ang
        self.thrusters_action_wind = [[] for _ in range(len(self.thrusters))]
        self.control_function = self.__default_control

    def update(self, control, low_step):
        if low_step is not None:
            self.dynamics.dynamic_model.dt = low_step
            self.dynamics.dynamic_model.h_old = low_step
        for thr_i in self.thrusters:
            thr_i.set_step_time(self.dynamics.dynamic_model.dt)
        [thr_i.set_ignition(control[i]) for i, thr_i in enumerate(self.thrusters)]
        [thr_i.propagate_thrust() for thr_i in self.thrusters]
        [thr_i.log_value() for thr_i in self.thrusters]

        thr_mag = np.array([thr_i.get_current_thrust() for thr_i in self.thrusters])
        m_dot_p = np.sum(np.array([thr_i.get_current_m_flow() for thr_i in self.thrusters]))

        thr_vec, tau_b = self.calc_thrust_torques(thr_mag)

        tau_ctrl = self.get_control_torque(self.dynamics.dynamic_model.current_pos_i,
                                           self.dynamics.dynamic_model.current_vel_i,
                                           self.dynamics.dynamic_model.current_theta,
                                           self.dynamics.dynamic_model.current_omega)
        tau_b = np.sum(tau_b) + tau_ctrl
        self.dynamics.dynamic_model.update(np.sum(thr_vec), m_dot_p, tau_ctrl, low_step)

    def calc_thrust_torques(self, thr_list):
        thr_vec = [thr_i * np.array([-np.sin(alpha_), np.cos(alpha_)])
                   for thr_i, alpha_ in zip(thr_list, self.thruster_ang)]

        tau_b = [np.cross(pos_i,thr_i) for pos_i, thr_i in zip(self.thruster_pos, thr_vec)]
        return np.array(tau_b), np.array(thr_vec)

    def get_control_torque(self, r, v, theta, omega):
        u_target = - v / np.linalg.norm(v)
        altitude = np.linalg.norm(r) - rm

        if altitude > 2000:
            u_current = np.array([-np.sin(theta + self.dynamics.dynamic_model.delta_alpha_k),
                                  np.cos(theta + self.dynamics.dynamic_model.delta_alpha_k)])
            ang_target = np.arccos(np.dot(u_target, u_current)) * np.sign(np.cross(u_current, u_target))
            omega_target = np.linalg.norm(v) / np.linalg.norm(r)
            p_gain = 0.01
            d_gain = 0.01
        else:
            ang_target = np.arctan2(r[0], r[1])
            p_gain = 0.00001
            d_gain = 0.00001
            omega_target = 0
        return p_gain * ang_target + d_gain * (omega_target - omega)

    def save_log(self):
        self.dynamics.dynamic_model.save_data()
        [thr_i.log_value() for thr_i in self.thrusters]

    def get_thrust(self):
        return np.sum(np.array([thr_i.current_mag_thrust_c for thr_i in self.thrusters]))

    def simulate(self, tf, low_step: float = None, progress: bool = True):
        # save ignition time and stop time
        subk = 0
        k = 0
        control = [0.0] * len(self.thrusters)
        while self.dynamics.dynamic_model.current_time <= tf and not self.dynamics.isTouchdown() and not self.dynamics.notMass():
            low_step_flag = False
            for i, thr in enumerate(self.thrusters):
                control[i] = self.control_function(self.dynamics.get_current_state(), n_e=i)
                if control[i] == 1 and not self.thrusters[i].thr_is_burned:
                    low_step_flag = sum([True, low_step_flag])
                    self.thrusters_action_wind[i].append(subk) if len(self.thrusters_action_wind[i]) == 0 else None
                elif self.thrusters[i].current_beta == 1 and self.thrusters[i].thr_is_burned is False:
                    low_step_flag = sum([True, low_step_flag])
                else:
                    self.thrusters_action_wind[i].append(subk) if len(self.thrusters_action_wind[i]) == 1 else None
                    low_step_flag = sum([False, low_step_flag])
                if (np.linalg.norm(self.dynamics.dynamic_model.current_pos_i) - 1.738e6) < 10e3:
                    low_step_flag = sum([True, low_step_flag])
                    low_step = 0.1
            subk += 1
            self.update(control, low_step if low_step_flag else None)
            left_engine = np.all(np.array([thr.thr_is_burned for thr in self.thrusters]))
            if left_engine:
                tf_update = self.get_orbit_period(tf)
                if tf_update + self.dynamics.dynamic_model.current_time < tf:
                    tf = tf_update + self.dynamics.dynamic_model.current_time
            self.save_log()
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

    def get_orbit_period(self, ctf):
        state = self.dynamics.get_current_state()
        mu = self.dynamics.mu
        r = state[0]
        v = state[1]
        energy = 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)
        a = - mu / energy / 2
        if a > rm:
            period = 2 * np.pi * np.sqrt(a ** 3 / mu)
            return period
        else:
            return ctf

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
        self.thrusters = []
        for i, value in enumerate(thrust_design):
            self.propellant_properties[i]['geometry']['setting']['ext_diameter'] = value
            self.thruster_conf[i]['case_diameter'] = value
            self.thrusters.append(Thruster(self.dynamics.dynamic_model.dt,
                                           self.thruster_conf[i],
                                           self.propellant_properties[i])())

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
        return np.sum(np.array([thr_i.channels['mass'].getPoint(0) for thr_i in self.thrusters]))
