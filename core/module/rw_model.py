"""
Created by Elias Obreque
Date: 05-12-2023
email: els.obrq@gmail.com
"""

import numpy as np
from tools.mathtools import runge_kutta_4


class RWModel(object):
    def __init__(self, rw_inertia, dt):
        self.inertia = rw_inertia
        self.dt = dt
        self.historical_rw_torque = [0]
        self.historical_rw_velocity = [0]
        self.target_angular_velocity = 0.0
        self.lag_coef = 0.2
        self.current_velocity = 0.0
        self.max_torque = 10
        self.min_torque = 0
        self.target_angular_accl_before = 0

    def dynamic_rw(self, state, t):
        rhs_ = (self.target_angular_velocity - state) / self.lag_coef
        return rhs_

    def propagate_rw(self):
        pre_angular_velocity = self.current_velocity
        self.target_angular_velocity = pre_angular_velocity + self.target_angular_accl_before * self.dt
        # propagation
        angular_velocity = pre_angular_velocity + runge_kutta_4(self.dynamic_rw, self.current_velocity, self.dt, None)

        self.current_velocity = angular_velocity
        angular_acc = (angular_velocity - pre_angular_velocity) / self.dt
        rw_torque = self.inertia * angular_acc
        self.historical_rw_torque.append(rw_torque)
        self.historical_rw_velocity.append(angular_velocity)
        return rw_torque

    def set_step_time(self, value):
        self.dt = value

    def set_torque(self, torque):
        ctrl_cycle = self.dt
        sign = 1
        if torque < 0:
            sign = -1

        if abs(torque) < self.max_torque:
            angular_acc = torque / self.inertia
        else:
            angular_acc = sign * self.max_torque / self.inertia
        if abs(torque) < self.min_torque:
            angular_acc = 0
        self.target_angular_accl_before = angular_acc


if __name__ == '__main__':
    pass
