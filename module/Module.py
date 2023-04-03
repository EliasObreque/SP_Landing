"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np

from dynamics.Dynamics import Dynamics
from thrust.thruster import Thruster


class Module(object):

    def __init__(self, mass, inertia, init_state, thruster_pos, thruster_ang, thruster_conf,
                 propellant_properties, reference_frame, dt):

        self.thrusters = [Thruster(dt, thruster_conf[i], propellant_properties[i])() for i in range(len(thruster_conf))]
        self.dynamics = Dynamics(dt, mass, inertia, init_state, reference_frame)
        self.thruster_pos = thruster_pos
        self.thruster_ang = thruster_ang
        self.thrusters_action_wind = [[]] * len(self.thrusters)
        # self.current_time = 0.0
        # self.dt = dt

    def update(self, control, low_step):
        if low_step is not None:
            self.dynamics.dynamic_model.dt = low_step
            self.dynamics.dynamic_model.h_old = low_step
        for thr_i in self.thrusters:
            thr_i.step_width = self.dynamics.dynamic_model.dt
        [thr_i.set_ignition(control[i]) for i, thr_i in enumerate(self.thrusters)]
        [thr_i.propagate_thrust() for thr_i in self.thrusters]
        # [thr_i.log_value() for thr_i in self.thrusters]
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

    def simulate(self, tf, low_step=None):
        # save ignition time and stop time
        subk = 0
        k = 0
        control = [0.0] * len(self.thrusters)
        while self.dynamics.dynamic_model.current_time <= tf and self.dynamics.isTouchdown() is False:
            for i, thr in enumerate(self.thrusters):
                control[i] = self.control_function(self.dynamics.get_current_state())
                if control[i] == 1 and self.thrusters[i].thr_is_burned == False:
                    self.thrusters_action_wind[i].append(subk) if len(self.thrusters_action_wind[i]) == 0 else None
                else:
                    self.thrusters_action_wind[i].append(subk) if len(self.thrusters_action_wind[i]) == 1 else None
                self.update(control, low_step)
                self.save_log()
                subk += 1
            if k > 29:
                print('Progress {} % - Thrust: {}'.format(self.dynamics.dynamic_model.current_time / tf * 100, self.get_thrust()))
                k = 0
            k += 1
        return self.dynamics.get_current_state()

    def train(self):
        pass

    def evaluate(self):
        pass

    @staticmethod
    def control_function(state):
        if state[0][0] >= 100:
            control = 1
        else:
            control = 0
        return control

    def set_control_function(self, function):
        self.control_function = function

    def get_ignition_state(self, moment):
        return self.thrusters_action_wind
