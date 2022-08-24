"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
from Dynamics.Dynamics import Dynamics
from Thrust.Thruster import Thruster


class Module(object):
    mu = 4.9048695e12  # m3s-2
    rm = 1.738e6

    def __init__(self, mass, inertia, n_thrusters, thruster_pos, thruster_ang, thruster_conf, propellant_properties,
                 reference_frame, dt):
        self.dynamics = Dynamics(dt, self.mu, self.rm, mass, reference_frame)
        self.thusters = [Thruster(dt, thruster_conf, propellant_properties, burn_type=None) for i in range(n_thrusters)]

    def update(self):
        self.dynamics.dynamic_model.rungeonestep()
        return

