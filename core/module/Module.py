"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
from core.dynamics.Dynamics import Dynamics
from tools.mathtools import propagate_rv_by_ang
from core.thrust.thruster import Thruster
from .pid import PID
from .rw_model import RWModel

rm = 1.738e6    # m
# EDL
ENTRY_MODE = 0
DESCENT_MODE = 1
LANDING_MODE = 2


class Module(object):
    thruster_conf = None
    propellant_properties = None

    def __init__(self, mass, inertia, init_state, sigma_r, sigma_v, thruster_pos, thruster_ang, thruster_conf,
                 propellant_properties, reference_frame, dt, training=False):
        self.mode = ENTRY_MODE
        self.thr_ignition_control = None
        self.training = training
        self.thruster_conf = thruster_conf
        self.propellant_properties = propellant_properties
        self.thrusters = [Thruster(dt, thruster_conf[i], propellant_properties[i])() for i in range(len(thruster_conf))]
        self.dynamics = Dynamics(dt, mass, inertia, init_state, reference_frame)
        self.thruster_pos = thruster_pos
        self.thruster_ang = thruster_ang
        self.thrusters_action_wind = [[] for _ in range(len(self.thrusters))]
        self.control_function = self.__default_control
        self.sigma_r, self.sigma_v = sigma_r, sigma_v
        self.sigma_theta = 0.0
        self.rev_count = 0.0
        self.historical_theta_error = []
        self.historical_omega_error = []
        self.control_pid = PID(1, 1, 1, dt)
        self.rw_model = RWModel(0.0009625, dt)
        self.dataset = []
        self.get_control_torque(self.dynamics.dynamic_model.current_pos_i,
                                self.dynamics.dynamic_model.current_vel_i,
                                self.dynamics.dynamic_model.current_theta,
                                self.dynamics.dynamic_model.current_omega)

    def propagate(self, tau_ctrl, control, low_step):
        if low_step is not None:
            self.dynamics.dynamic_model.dt = low_step
            self.dynamics.dynamic_model.h_old = low_step
            self.control_pid.set_step_time(low_step)
            self.rw_model.set_step_time(low_step)
        for thr_i in self.thrusters:
            thr_i.set_step_time(self.dynamics.dynamic_model.dt)
        [thr_i.set_ignition(control[i]) for i, thr_i in enumerate(self.thrusters)]
        [thr_i.propagate_thrust() for thr_i in self.thrusters]

        thr_mag = np.array([thr_i.get_current_thrust() for thr_i in self.thrusters])
        m_dot_p = np.sum(np.array([thr_i.get_current_m_flow() for thr_i in self.thrusters]))

        tau_b, thr_vec = self.calc_thrust_torques(thr_mag)
        self.rw_model.set_torque(tau_ctrl)
        new_tau = self.rw_model.propagate_rw()
        tau_b = tau_b + new_tau
        self.dynamics.dynamic_model.propagate(thr_vec, m_dot_p, tau_b, low_step)

    def calc_thrust_torques(self, thr_list):
        thr_vec = [thr_i * np.array([-np.sin(alpha_), np.cos(alpha_)])
                   for thr_i, alpha_ in zip(thr_list, self.thruster_ang)]

        tau_b = [np.cross(pos_i, thr_i) for pos_i, thr_i in zip(self.thruster_pos, thr_vec)]
        return np.sum(np.array(tau_b)), np.sum(np.array(thr_vec), axis=0)

    def get_control_torque(self, r, v, theta, omega):
        u_target = - v / np.linalg.norm(v)
        altitude = np.linalg.norm(r) - rm
        u_current = np.array([-np.sin(theta),
                              np.cos(theta)])
        dot_vec = np.dot(u_target, u_current)
        if abs(dot_vec) > 1:
            dot_vec = 1 * np.sign(dot_vec)
        ang_error = np.arccos(dot_vec) * np.sign(np.cross(u_current, u_target))
        if self.mode == ENTRY_MODE:
            omega_target = np.linalg.norm(v) / np.linalg.norm(r)
            p_gain = 1e-2
            d_gain = 0.0
            i_gain = 1e-1
            self.control_pid.set_mode(ENTRY_MODE)
        elif self.mode == DESCENT_MODE:
            self.control_pid.set_mode(DESCENT_MODE)
            p_gain = 1e-2
            d_gain = 0.0
            i_gain = 1e-1
            omega_target = 0
        else:
            self.control_pid.set_mode(LANDING_MODE)
            p_gain = 1e-2
            d_gain = 0.0
            i_gain = 1e-1
            omega_target = 0

        self.historical_theta_error.append(ang_error)
        self.historical_omega_error.append(omega_target - omega)
        self.control_pid.set_gain(p_gain, d_gain, i_gain)
        ctrl = self.control_pid.calc_control(ang_error, omega_target - omega, 2)
        return ctrl

    def save_log(self):
        self.dynamics.dynamic_model.save_data()
        [thr_i.log_value() for thr_i in self.thrusters]

    def get_thrust(self):
        return np.sum(np.array([thr_i.current_mag_thrust_c for thr_i in self.thrusters]))

    def get_control(self, pos_, n_e=0):
        if self.mode == ENTRY_MODE:
            ctrl = self.on_off_control_by_ang(pos_, n_e)
        elif self.mode == DESCENT_MODE or self.mode == LANDING_MODE:
            ctrl = self.on_off_control_by_alt(pos_, n_e)
        else:
            ctrl = 0
        return ctrl

    def is_contact_at_ignition(self):
        if self.mode == ENTRY_MODE:
            pos = self.dynamics.dynamic_model.current_pos_i
            value = np.arctan2(pos[1], pos[0])
            if value < 0:
                value += 2 * np.pi
            return abs(self.thr_ignition_control[0] - value) < 5e-3
        elif self.mode == DESCENT_MODE or self.mode == LANDING_MODE:
            ctrl = False
        else:
            ctrl = False
        return ctrl

    def simulate(self, tf, low_step: float = None, progress: bool = True, only_thrust: bool = False,
                 force_step: bool = False, force_mode: int = None):
        # save ignition time and stop time
        subk = 0
        k = 0
        ignition_control = [0.0 for _ in range(len(self.thrusters))]
        while self.dynamics.dynamic_model.current_time <= tf and not self.dynamics.isTouchdown() and not self.dynamics.notMass():
            low_step_flag = False

            # Estimation
            alt_nom = np.linalg.norm(self.dynamics.get_current_state()[0]) - rm
            if alt_nom < 2000:
                self.sigma_r = 0.1
                self.sigma_v = 0.5
            pos_ = self.dynamics.get_current_state()[0] + np.random.normal(0, (self.sigma_r, self.sigma_r))
            vel_ = self.dynamics.get_current_state()[1] + np.random.normal(0, (self.sigma_v, self.sigma_v))
            theta = self.dynamics.get_current_state()[2] + np.random.normal(0, (self.sigma_theta,
                                                                                self.sigma_theta))
            alt = np.linalg.norm(pos_) - rm
            # engines states
            array_state = np.array([thr.thr_was_burned for thr in self.thrusters])
            if force_mode is None:
                if not np.all(array_state[:2]):
                    self.mode = ENTRY_MODE
                else:
                    self.mode = DESCENT_MODE
            else:
                self.mode = force_mode
            # control
            # low step by time
            is_contact = self.is_contact_at_ignition()
            ignition_control = [self.get_control(pos_, n_e=i) for i in range(len(self.thrusters))]

            # low step by burning
            is_burning = np.any(np.array([thr.current_beta for thr in self.thrusters]))
            tau_control = self.get_control_torque(self.dynamics.dynamic_model.current_pos_i,
                                                  self.dynamics.dynamic_model.current_vel_i,
                                                  self.dynamics.dynamic_model.current_theta,
                                                  self.dynamics.dynamic_model.current_omega)
            # low step  by altitude
            is_low_altitude = False
            if (np.linalg.norm(pos_) - 1.738e6) < 200e3:
                is_low_altitude = True
                low_step = 0.1
            if (np.linalg.norm(pos_) - 1.738e6) < 15e3:
                is_low_altitude = True
                low_step = 0.01

            # # low step by attitude
            is_low_attitude = False
            # if np.abs(self.dynamics.dynamic_model.current_omega) > 1e-4:
            #     is_low_attitude = True
            low_step_flag = is_contact or force_step or is_burning or is_low_altitude or is_low_attitude

            if not low_step_flag:
                tau_control = 0.0
                self.control_pid.error_int = 0
                self.control_pid.last_error = 0
            low_step_ = low_step if low_step_flag else None
            subk += 1
            # saving activation points
            for i, thr in enumerate(self.thrusters):
                if ignition_control[i] == 1 and not self.thrusters[i].thr_was_burned:
                    self.thrusters_action_wind[i].append(subk) if len(self.thrusters_action_wind[i]) == 0 else None
                else:
                    self.thrusters_action_wind[i].append(subk) if len(self.thrusters_action_wind[i]) == 1 else None

            # propagation
            self.propagate(tau_control, ignition_control, low_step_)
            left_engine = np.all(array_state)
            if left_engine:
                tf_update = self.get_orbit_period(tf)
                if tf_update + self.dynamics.dynamic_model.current_time < tf:
                    tf = tf_update + self.dynamics.dynamic_model.current_time
            self.save_log()
            if tf < self.dynamics.dynamic_model.current_time:
                break
            if only_thrust:
                if left_engine:
                    break
            if k > 29 and progress:
                print('Progress {} % - Thrust: {}'.format(self.dynamics.dynamic_model.current_time / tf * 100,
                                                          self.get_thrust()))
                k = 0
            k += 1
        self.dataset = self.dynamics.dynamic_model.get_historial()
        self.dataset.insert(-1, self.get_betas_control())
        return self.dataset

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

    def __default_control(self, state, n_e=0):
        alt = np.linalg.norm(state[0]) - rm
        if alt * 1e-3 < self.thr_ignition_control[n_e]:
            return 1
        else:
            return 0

    def on_off_control_by_alt(self, state, n_e=0):
        alt = np.linalg.norm(state) - rm
        if alt < self.thr_ignition_control[n_e]:
            return 1
        else:
            return 0

    def on_off_control_by_ang(self, value_vec, n_e=0):
        pos = value_vec
        alt = np.linalg.norm(pos) - rm
        if n_e == 0 or n_e == 1:
            value = np.arctan2(pos[1], pos[0])
            if value < 0:
                value += 2 * np.pi
            if value >= self.thr_ignition_control[n_e] and alt > 20000e3:
                return 1
            else:
                return 0
        elif n_e == 2:
            value = np.arctan2(pos[1], pos[0])
            if value < 0:
                value += 2 * np.pi
            if value >= self.thr_ignition_control[n_e]:
                return 1
            else:
                return 0
        elif n_e == 3 or n_e == 4:
            value = np.arctan2(pos[1], pos[0])
            if value < 0:
                value += 2 * np.pi
            if value >= self.thr_ignition_control[n_e]:
                return 1
            else:
                return 0
        elif n_e > 4:
            value = np.linalg.norm(pos) - rm
            if value < self.thr_ignition_control[n_e]:
                return 1
            else:
                return 0
        else:
            value = np.linalg.norm(value_vec[0])
            if value < self.thr_ignition_control[n_e]:
                return 1
            else:
                return 0

    def set_control_function(self, control_parameter, default=False):
        self.thr_ignition_control = control_parameter
        if not default:
            self.control_function = self.on_off_control_by_ang

    def set_thrust_design(self, thrust_diameter, thrust_large=None, **kwargs):
        self.thrusters = []
        for i, value in enumerate(thrust_diameter):
            self.propellant_properties[i]['geometry']['setting']['ext_diameter'] = value
            if 'int_diameter' in list(kwargs.keys()):
                self.propellant_properties[i]['geometry']['setting']['int_diameter'] = kwargs['int_diameter'][i]
            self.thruster_conf[i]['case_diameter'] = value
            if thrust_large is not None:
                self.propellant_properties[i]['geometry']['setting']['large'] = thrust_large[i]
                self.thruster_conf[i]['case_large'] = thrust_large[i]
            self.thrusters.append(Thruster(self.dynamics.dynamic_model.dt,
                                           self.thruster_conf[i],
                                           self.propellant_properties[i])())
            if "kn" in list(kwargs.keys()):
                self.thrusters[i].set_kn(kwargs["kn"])

    def set_thrust_bias(self, bias_isp: list, dead_time: list = None):
        for i, value in enumerate(bias_isp):
            self.thrusters[i].set_bias_isp(value)
            if dead_time is not None:
                self.thrusters[i].set_dead_time(dead_time[i])

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

    def get_mass_burned(self):
        return np.sum(np.array([thr_i.channels['mass'].getLast() for thr_i in self.thrusters]))

    def get_betas_control(self):
        betas = [thr.historical_beta for thr in self.thrusters]
        return np.array(betas).T

