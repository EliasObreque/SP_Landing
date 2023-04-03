"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
engine_diameter_ext = 0.015
throat_diameter = 0.005 # m
e_nozzle_diameter = 0.01  # m
height = 0.15  # m
file_name = "thrust/StarGrain7.csv"

NEUTRAL = 'neutral'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'

FILE = 'file'
MODEL = 'model'
GRAIN = 'grain'

# thrust profile
# if FILE:
by_file = {'type': FILE, 'file_name': 'thrust/dataThrust/5kgEngine.csv', 'isp': 200, 'dt': 0.1,
           'ThrustName': 'Thrust(N)', 'TimeName': 'Time(s)'} # Change these parameters
# if MODEL:
by_model = {'type': MODEL, 'performance': {'t_burn': 7.5,
                                           'max_mass_flow': 1.9,
                                           'cross_section': PROGRESSIVE,
                                           'isp': 210,
                                           'isp_noise_std': None,
                                           'isp_bias_std': None}}
# if GRAIN:
by_grain = {'type': GRAIN}


default_thruster = {'throat_diameter': throat_diameter,
                    'case_diameter': engine_diameter_ext,
                    'case_large': height,
                    'exit_nozzle_diameter': e_nozzle_diameter,
                    'convergent_angle_deg': 60,
                    'divergent_angle_deg': 15,
                    'thrust_profile': by_grain,
                    'max_ignition_dead_time': 0.5}
