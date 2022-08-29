"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np

from Dynamics.Dynamics import Dynamics
from thrust.thruster import Thruster


class Module(object):

    def __init__(self, mass, inertia, init_state, n_thrusters, thruster_pos, thruster_ang, thruster_conf,
                 propellant_properties, reference_frame, dt):

        self.thrusters = [Thruster(dt, thruster_conf[i], propellant_properties[i], burn_type=None) for i in range(n_thrusters)]
        self.dynamics = Dynamics(dt, mass, inertia, init_state, mass, reference_frame)
        self.thruster_pos = thruster_pos
        self.thruster_ang = thruster_ang

    def update(self, control):
        [thr_i.set_beta(control) for thr_i in self.thrusters]
        [thr_i.propagate_thr() for thr_i in self.thrusters]
        thr = [thr_i.get_current_thrust() for thr_i in self.thrusters]
        m_dot_p = [thr_i.get_current_m_flow() for thr_i in self.thrusters]
        tau_b = self.calc_torques(thr)
        self.dynamics.dynamic_model.update(thr, tau_b)
        return

    def calc_torques(self, thr_list):
        tau_b = [np.cross(pos_i, thr_i) for pos_i, thr_i in (self.thruster_pos, thr_list)]
        return tau_b
