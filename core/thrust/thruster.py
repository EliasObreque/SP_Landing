"""
Created: 7/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
import numpy as np
from core.thrust.dataThrust.thrustData import ThrustMeasure
from core.thrust.engine.engine import Engine
from core.thrust.model.thrustModel import MathModel
from core.thrust.thrustProperties import MODEL, GRAIN, FILE
from abc import ABC

DEG2RAD = np.pi/180
ge = 9.807


class Thruster(ABC):
    def __init__(self, dt, thruster_properties, propellant_properties):
        if thruster_properties['thrust_profile']['type'] == MODEL:
            self.base = MathModel(dt, thruster_properties)
        elif thruster_properties['thrust_profile']['type'] == FILE:
            self.base = ThrustMeasure(thruster_properties)
        elif thruster_properties['thrust_profile']['type'] == GRAIN:
            self.base = Engine(dt, thruster_properties, propellant_properties)
        else:
            print('Error selecting thrust profile simulation')

    def __call__(self, *args, **kwargs):
        return self.base

    def get_time(self):
        self.base.get_time()


if __name__ == '__main__':
    from thrustProperties import default_thruster, second_thruster, main_thruster
    from tools.Viewer import plot_thrust, show_plot
    from core.thrust.propellant.source.propellant_data import propellant_data
    from propellant.propellantProperties import default_propellant, BATES, bates_geom, main_propellant, second_propellant
    import matplotlib.pyplot as plt

    NEUTRAL = 'neutral'
    PROGRESSIVE = 'progressive'
    REGRESSIVE = 'regressive'
    m0 = 24
    ge = 9.807
    dt = 0.01

    file_name = "thrust/StarGrain7.csv"

    mixture_name = 'RCS - Blue Thunder'
    propellant_data_ = [pro_data for pro_data in propellant_data if pro_data['name'] == mixture_name][0]
    Isp = propellant_data_['data']['Isp']
    propellant_properties_ = main_propellant
    propellant_properties_['mixture_name'] = mixture_name
    # propellant_properties_['geometry']['type'] = BATES
    # if propellant_properties_['geometry']['type'] is not None:
    #   propellant_properties_['geometry']['setting'] = bates_geom

    percentage_variation_n = 0
    upper_isp_noise = Isp * (1.0 + percentage_variation_n / 100.0)
    propellant_properties_['isp_noise_std'] = (upper_isp_noise - Isp) / 3

    percentage_variation_b = 0
    upper_isp_bias = Isp * (1.0 + percentage_variation_b / 100.0)
    propellant_properties_['isp_bias_std'] = (upper_isp_bias - Isp) / 3

    thruster_properties_ = main_thruster
    thruster_properties_['thrust_profile'] = {'type': GRAIN}
    # thruster_properties_['thrust_profile']['type'] = MODEL
    thruster_properties_['max_ignition_dead_time'] = 0.0
    ctrl_a = [1.0]
    ctrl_b = [6.91036]
    max_mass_flow = 1 / Isp / ge
    t_burn = 20
    json_list = {'1': {'Best_individual': [max_mass_flow, t_burn, ctrl_a, ctrl_b]}}

    if thruster_properties_['thrust_profile']['type'] == MODEL:
        thruster_properties_['thrust_profile']['performance']['cross_section'] = REGRESSIVE
        thruster_properties_['thrust_profile']['performance']['isp'] = Isp
        thruster_properties_['thrust_profile']['performance']['isp_noise_std'] = (upper_isp_noise - Isp) / 3
        thruster_properties_['thrust_profile']['performance']['isp_bias_std'] = (upper_isp_bias - Isp) / 3
        thruster_properties_['thrust_profile']['performance']['t_burn'] = t_burn
        thruster_properties_['thrust_profile']['performance']['max_mass_flow'] = max_mass_flow

    n_thruster = 1
    comp_thrust = []
    for i in range(n_thruster):
        comp_thrust.append(Thruster(dt, thruster_properties_, propellant_properties_)())

    # IDEAL CASE
    propellant_properties_['isp_noise_std'] = None
    propellant_properties_['isp_bias_std'] = None

    if thruster_properties_['thrust_profile']['type'] == MODEL:
        thruster_properties_['thrust_profile']['performance']['isp_noise_std'] = 0.0
        thruster_properties_['thrust_profile']['performance']['isp_bias_std'] = 0.0

    thruster_properties_['max_ignition_dead_time'] = 0.0

    comp_thrust_free = Thruster(dt, thruster_properties_, propellant_properties_)()

    time_array = [0]
    k = 1
    current_time = 0
    beta = 0
    while current_time <= 2 * t_burn:
        thr = 0
        if current_time >= 0.5:
            beta = 1
        for i in range(n_thruster):
            comp_thrust[i].set_ignition(beta)
            comp_thrust[i].propagate_thrust()
            comp_thrust[i].log_value()

        comp_thrust_free.set_ignition(beta)
        comp_thrust_free.propagate_thrust()
        comp_thrust_free.log_value()
        current_time += dt
        time_array.append(current_time)

    mass = comp_thrust_free.channels['mass'].getData()
    mass_flow = comp_thrust_free.channels['massFlow'].getData()
    print(dt * np.sum(mass_flow), max(mass))
    total_thrust = 0
    torque = 0
    total_kn = 0
    total_pressure = 0
    total_web = 0
    for hist in comp_thrust:
        total_thrust += np.array(hist.historical_mag_thrust)
        total_kn += hist.channels['kn'].getData()
        total_web += hist.channels['web'].getData()
        total_pressure += hist.channels['pressure'].getData() * 1e-6
    # print([elem[1].getData() for elem in comp_thrust[0]().channels.items()])
    # print("radius mm: ", comp_thrust[0].calc_area_by_mass_flow(10.16e-3))

    plt.figure()
    plt.grid()
    plt.title("Propellant Mass [kg]")
    plt.plot(time_array, mass)

    plt.figure()
    plt.grid()
    plt.title("Mass flow [kg/s]")
    plt.plot(time_array, mass_flow)

    plt.figure()
    plt.grid()
    plt.title("KN ")
    plt.plot(time_array, total_kn)

    plt.figure()
    plt.grid()
    plt.title("Web [mm]")
    plt.plot(time_array, total_web * 1e3)

    plt.figure()
    plt.grid()
    plt.title("Chamber Pressure [MPa]")
    plt.plot(time_array, total_pressure)

    plot_thrust(time_array, total_thrust, thrust_free=comp_thrust_free.historical_mag_thrust,
                names=['Model thrust with dead time', 'Ideal thrust'], dead=.0)
    show_plot()
