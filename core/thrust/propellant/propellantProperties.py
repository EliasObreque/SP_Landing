"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'
TUBREG = 'tub_reg'
CUSTOM = 'custom'

# inhibit: True 1, False 0
name_prop = 'RCS - Blue Thunder'

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

tubular_geom = {'ext_diameter': 0.035,
                'int_diameter': None,
                'large': 0.1,
                'inhibit': {'top': 0, 'bot': 1}}

tubreg_geom = {'ext_diameter': 0.15,
                'int_diameter': 0.05,
                'large': 0.2,
                'inhibit': {'top': 0, 'bot': 1}}

custom_geom = {'ext_diameter': 0.1,
               'int_diameter': 0.02,
               'large': 0.2,
               'inhibit': {'top': 1, 'bot': 1}}

default_propellant = {'mixture_name': name_prop,
                      'geometry': {'type': BATES,
                                   'setting': bates_geom},
                      'isp_noise_std': 3.25,
                      'isp_bias_std': 0.0}

main_propellant = {'mixture_name': name_prop,
                   'geometry': {'type': TUBULAR,
                                'setting': tubular_geom},
                   'isp_noise_std': 0,
                   'isp_bias_std': 0.0}

second_propellant = {'mixture_name': name_prop,
                     'geometry': {'type': TUBULAR,
                                  'setting': tubular_geom},
                     'isp_noise_std': 0.0,
                     'isp_bias_std': 0.0}

bates2_geom = {'ext_diameter': 0.03,
               'int_diameter': 0.015,
               'large': 0.2,
               'inhibit': {'top': 1, 'bot': 1}}

tubular2_geom = {'ext_diameter': 0.03,
                 'int_diameter': None,
                 'large': 0.2,
                 'inhibit': {'top': 0, 'bot': 1}}

third_propellant = {'mixture_name': name_prop,
                    'geometry': {'type': TUBULAR,
                                 'setting': tubular2_geom},
                    'isp_noise_std': 3.25,
                    'isp_bias_std': 0.0}
