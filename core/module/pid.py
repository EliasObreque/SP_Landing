
import numpy as np
REF_POINT = 2
NAD_POINT = 1
DETUMBLING = 0


class PID(object):
    def __init__(self, P, I, D, step, ctrl_min=None, ctrl_max=None):
        """
        :param P: Proportional gain, 3x3 Numpy diagonal matrix.
        :param I: Integral gain, 3x3 Numpy diagonal matrix.
        :param D: Derivative gain, 3x3 Numpy diagonal matrix
        :param step: Step time for propagation of control signal, float.
        :param ctrl_min: Inferior limits for control signal, 3D Numpy array.
        :param ctrl_max: Superior limits for control signal, 3D Numpy array.
        """

        self.P = P
        self.I = I
        self.D = D
        self.error_int = 0.0
        self.step_width = step
        self.last_error = 0
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max

    def set_step_time(self, dt):
        self.step_width = dt

    def set_gain(self, newP, newI, newD):
        """
        This method sets new gains in an already created controller.

        :param newP: New P gain, 3x3 Numpy diagonal matrix.
        :param newI: New I gain, 3x3 Numpy diagonal matrix.
        :param newD: New D gain, 3x3 Numpy diagonal matrix.
        """

        self.P = newP
        self.I = newI
        self.D = newD

    def calc_control(self, error_quat, error_omega, type_control):
        """
        This method calculates the control signal to use as input in the plant.

        :param error_quat: Error between calculated/measured/estimated quaternion and the reference, 3D Numpy array.
        :param error_omega: Error between calculated/measured/estimated angular velocity and the reference, 3D Numpy array.
        :param type_control: Select mode used in the ADCS. Can be DETUMBLING, NAD_POINT or REF_POINT.
        :return: Control signal, with or without antiwinding up, 3D Numpy array.
        """

        if type_control == DETUMBLING:
            error = error_omega
            error_diff = (error - self.last_error)/self.step_width
            self.error_int += error*self.step_width
            self.last_error = error

        elif type_control == NAD_POINT:
            error = error_quat
            error_diff = error_omega
            self.error_int += error_quat* self.step_width

        elif type_control == REF_POINT:
            error = error_quat
            error_diff = error_omega
            self.error_int += error_quat * self.step_width

        else:
            error = 0
            error_diff = 0
            self.error_int = 0

        ctrl = self.P * error + self.I * self.error_int + self.D * error_diff
        return ctrl
