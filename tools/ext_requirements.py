"""
Created by:

@author: Elias Obreque
@Date: 1/7/2021 12:20 AM 
els.obrq@gmail.com

"""
import numpy as np
import time
import sys
import codecs
import json
import os

if os.path.isdir("./logs/") is False:
    os.mkdir("./logs/")


def velocity_req(vp, va, r_planet, mu, periaxis, apoaxis):
    # Falling speed required
    vfp = np.sqrt(2 * mu / periaxis * (1 - r_planet / periaxis))
    vfa = np.sqrt(2 * mu / apoaxis * (1 - r_planet / apoaxis))
    dv_req_p = vp + vfp
    dv_req_a = va + vfa
    print('Accumulated velocity from perilune [m/s]: ', dv_req_p)
    print('Accumulated velocity from apolune [m/s]: ', dv_req_a)
    return dv_req_p, dv_req_a


def mass_req(dv_req, c_char, density_propellant, m0):
    # Mass required
    mass_ratio = np.exp(dv_req / c_char)
    m1 = m0 / mass_ratio
    mp = m0 - m1
    print('Required mass for propulsion: ', mp, ' [kg]')
    print('Required volume for propulsion: ', mp/density_propellant, ' [cc]')
    print('Available mass for payload, structure, and subsystems', m1, ' [kg]')
    return mp, m1


def save_data(master_data, folder_name, filename):
    """
    :param master_data: Dictionary
    :param folder_name: an Name with /, for example folder_name = "2000m/"
    :param filename: string name
    :return:
    """
    if os.path.isdir("./logs/" + folder_name) is False:
        temp_list = folder_name.split("/")
        fname = ''
        for i in range(len(temp_list) - 1):
            fname += temp_list[:i + 1][i]
            if os.path.isdir("./logs/" + fname) is False:
                os.mkdir("./logs/" + fname)
            fname += "/"
    with codecs.open("./logs/" + folder_name + filename + ".json", 'w') as file:
        json.dump(master_data, file)
    print("Data saved to file:", filename)