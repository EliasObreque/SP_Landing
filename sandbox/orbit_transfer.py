"""
Created by Elias Obreque
Date: 30-07-2023
email: els.obrq@gmail.com
"""

import numpy as np

mu = 4.9048695e12  # m3s-2
rm = 1.738e6
ra = 68e6
rp = 2e6
a = 0.5 * (ra + rp)


def get_energy(mu, r, v):
    return 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)


def tsiolkovsky_equation(delta_v, specific_impulse):
  ratio = np.exp(-delta_v / specific_impulse / 9.80665)
  return ratio


if __name__ == '__main__':
    vp = np.sqrt(2 * mu / rp - mu / a)
    at = 0.5 * (2e6 + 100e3)
    vt = np.sqrt(2 * mu / rp - mu / at)
    dv = vp - vt
    print(dv)
    mass_ratio = tsiolkovsky_equation(dv, 280)
    print("mass {}".format(24. * mass_ratio))

    mass_req = 24 - 24. * mass_ratio
    large = 0.2
    diam_int = 0.01
    dens = 1.66 * 1e3    # kg/m3
    area_int = np.pi * (diam_int * 0.5)**2
    area_ext = mass_req / dens / large - area_int
    rad_ext = np.sqrt(area_ext / np.pi)
    diam_ext = 2 * rad_ext
    print("mass req: {}, diameter ext: {}".format(mass_req, diam_ext))