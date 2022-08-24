"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np


class Engine(object):
    def __init__(self, dt, thruster_properties):
        self.step = dt
        self.throat_diameter = thruster_properties['throat_diameter']
        self.diameter_ext = thruster_properties['case_diameter']
        self.case_large = thruster_properties['case_large']
        self.divergent_angle = np.deg2rad(thruster_properties['divergent_angle_deg'])
        self.convergent_angle = np.deg2rad(thruster_properties['convergent_angle_deg'])
        self.exit_nozzle_diameter = thruster_properties['exit_nozzle_diameter']

        d = 0.5 * (self.exit_nozzle_diameter - self.throat_diameter) / np.tan(self.convergent_angle)
        volume_convergent_zone = (np.pi * d * (self.diameter_ext * 0.5) ** 2) / 3
        volume_case = np.pi * ((self.diameter_ext * 0.5) ** 2) * self.case_large
        self.volume = volume_case + volume_convergent_zone
