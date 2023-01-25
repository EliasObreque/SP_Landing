"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
engine_diameter_ext = 0.03
throat_diameter = 0.003  # m
e_nozzle_diameter = 0.01  # m
height = 0.2  # m
file_name = "thrust/StarGrain7.csv"

NEUTRAL = 'neutral'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'

FILE = 'file'
MODEL = 'model'
GRAIN = 'grain'

# thrust profile
# if FILE:
by_file = {'type': FILE, 'file_name': None, 'isp': 200}
# if MODEL:
by_model = {'type': MODEL, 'performance': {'t_burn': 10.0,
                                           'max_mass_flow': 0.1,
                                           'cross_section': REGRESSIVE,
                                           'isp': 200,
                                           'isp_noise_std': None,
                                           'isp_bias_std': None}}
# if GRAIN:
by_grain = {'type': GRAIN}


default_thruster = {'throat_diameter': throat_diameter,
                    'case_diameter': engine_diameter_ext,
                    'case_large': height,
                    'exit_nozzle_diameter': e_nozzle_diameter,
                    'convergent_angle_deg': 45,
                    'divergent_angle_deg': 15,
                    'thrust_profile': by_model,
                    'max_ignition_dead_time': 0.5}
