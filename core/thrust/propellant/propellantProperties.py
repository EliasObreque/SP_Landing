"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'
CUSTOM = 'custom'

# inhibit: True 1, False 0

bates_geom = {'ext_diameter': 0.15,
              'int_diameter': 0.03,
              'large': 0.2,
              'inhibit': {'top': 1, 'bot': 1}}

star_geom = {'ext_diameter': 0.03,
             'int_diameter': 0.005,
             'n_points': 5,
             'star_angle_deg': 10,
             'large': 0.2,
             'inhibit': {'top': 1, 'bot': 1}}

tubular_geom = {'ext_diameter': 0.061,
                'int_diameter': None,
                'large': 0.2,
                'inhibit': {'top': 0, 'bot': 1}}

custom_geom = {'ext_diameter': 0.1,
               'int_diameter': 0.02,
               'large': 0.2,
               'inhibit': {'top': 1, 'bot': 1}}

default_propellant = {'mixture_name': 'Nakka - KNSU',
                      'geometry': {'type': BATES,
                                   'setting': bates_geom},
                      'isp_noise_std': 3.25,
                      'isp_bias_std': 10.83}

main_propellant = {'mixture_name': 'Nakka - KNSU',
                   'geometry': {'type': BATES,
                                'setting': bates_geom},
                   'isp_noise_std': 3.25,
                   'isp_bias_std': 10.83}

second_propellant = {'mixture_name': 'Nakka - KNSU',
                     'geometry': {'type': TUBULAR,
                                  'setting': tubular_geom},
                     'isp_noise_std': 3.25,
                     'isp_bias_std': 10.83}
