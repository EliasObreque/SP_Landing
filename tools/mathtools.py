"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import numpy as np
import datetime

twopi = 2.0 * np.pi
deg2rad = np.pi / 180.0
a = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
b = np.array([[0, 0, 0, 0, 0],
              [1 / 4, 0, 0, 0, 0],
              [3 / 32, 9 / 32, 0, 0, 0],
              [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
              [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
              [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]])
c4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
c5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])


def runge_kutta_4(function, x, dt, ct, *args):
    k1 = function(x, ct, *args)
    xk2 = x + (dt / 2.0) * k1

    k2 = function(xk2, ct, *args)
    xk3 = x + (dt / 2.0) * k2

    k3 = function(xk3, ct, *args)
    xk4 = x + dt * k3

    k4 = function(xk4, ct, *args)

    next_x = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_x


def rkf45(ode_function, xi, h_old, t, tf, *args, tol=1e-9):
    hmin = 1e-3
    h = h_old
    k_i = np.zeros(len(xi))
    x_out = np.zeros(len(xi))
    k_nn = np.tile(k_i, (6, 1))
    end_condition = False
    while end_condition is False:
        for i in range(6):
            t_inner = t + a[i] * h
            x_inner = xi
            for j in range(i):
                x_inner = x_inner + h * k_nn[j, :] * b[i][j]
            k_nn[i, :] = ode_function(x_inner, t_inner, *args)
        te = h * np.dot(c4 - c5, k_nn)
        error = np.max(np.abs(te))
        xmax = np.max(np.abs(xi))
        te_allowed = tol * max(xmax, 1.0)
        delta = (te_allowed / (error + np.finfo(np.float64).eps)) ** (1.0 / 5.0)
        if error <= te_allowed:
            end_condition = True
            h = min(h, tf - t)
            x_out = xi + h * np.dot(c5, k_nn)
        h = min(delta * h, 4 * h)
        if h < hmin:
            raise print('Warning: Step size fell below its minimum allowable value {}'.format(h))
    return x_out, h


def jday(year, mon, day, hr, minute, sec):
    jd0 = 367.0 * year - 7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 + 275.0 * mon // 9.0 + day + 1721013.5
    utc = ((sec / 60.0 + minute) / 60.0 + hr)  # utc in hours#
    return jd0 + utc / 24.


def gstime(jdut1):
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
           (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841  # sec
    temp = (temp * deg2rad / 240.0) % twopi  # 360/86400 = 1/240, to deg, to rad

    #  ------------------------ check quadrants ---------------------
    if temp < 0.0:
        temp += twopi

    return temp


def fmod2(x):
    if x > np.pi:
        x -= twopi
    elif x < -np.pi:
        x += twopi
    else:
        x = x
    return x


def julian_to_datetime(julian_date):
    base_date = datetime.datetime(2000, 1, 1)  # Julian date 2451545 corresponds to this datetime
    days_elapsed = julian_date - 2451545
    delta = datetime.timedelta(days=days_elapsed)
    converted_datetime = base_date + delta
    return converted_datetime


def jd_to_decyear(jd):
    # --------------- find year and days of the year ---------------
    temp = jd - 2415019.5
    tu = temp / 365.25
    year = 1900 + np.floor(tu)
    leapyrs = np.floor((year - 1901) * 0.25)

    # optional nudge by 8.64x10-7 sec to get even outputs
    days = temp - ((year - 1900) * 365.0 + leapyrs) + 0.00000000001

    # ------------ check for case of beginning of a year -----------
    if days < 1.0:
        year = year - 1
        leapyrs = np.floor((year - 1901) * 0.25)
        days = temp - ((year - 1900) * 365.0 + leapyrs)

    decyear = year + days / 365.25
    return decyear


def timestamp_to_julian(timestamp):
    unix_epoch = datetime.datetime(1970, 1, 1)
    input_datetime = unix_epoch + datetime.timedelta(seconds=timestamp)
    julian_date = 2440587.5 + (input_datetime - unix_epoch).total_seconds() / 86400
    return julian_date


def tle_epoch_to_julian(tle_epoch):
    # Extrae el año y el día del año del valor de época
    year = int(tle_epoch[:2])
    day_of_year = int(tle_epoch[2:5])
    fraction_of_day = float("0" + tle_epoch[5:])
    # Calcula la fecha juliana
    base_date = jday(2000 + year, 1, 0,  0, 0, 0)
    julian_date = base_date + day_of_year + fraction_of_day
    return julian_date


def skewsymmetricmatrix(x_omega_b):
    S_omega = np.zeros((3, 3))
    S_omega[1, 0] = x_omega_b[2]
    S_omega[2, 0] = -x_omega_b[1]

    S_omega[0, 1] = -x_omega_b[2]
    S_omega[0, 2] = x_omega_b[1]

    S_omega[2, 1] = x_omega_b[0]
    S_omega[1, 2] = -x_omega_b[0]
    return S_omega


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def omega4kinematics(x_omega_b):
    Omega = np.zeros((4,4))
    Omega[1, 0] = -x_omega_b[2]
    Omega[2, 0] = x_omega_b[1]
    Omega[3, 0] = -x_omega_b[0]

    Omega[0, 1] = x_omega_b[2]
    Omega[0, 2] = -x_omega_b[1]
    Omega[0, 3] = x_omega_b[0]

    Omega[1, 2] = x_omega_b[0]
    Omega[1, 3] = x_omega_b[1]

    Omega[2, 1] = -x_omega_b[0]
    Omega[2, 3] = x_omega_b[2]

    Omega[3, 1] = -x_omega_b[1]
    Omega[3, 2] = -x_omega_b[2]
    return Omega


def get_mrp_from_q(q):
    p = q[:3]/(1 + q[3])
    if np.linalg.norm(p) > 1:
        p = -p / np.linalg.norm(p) ** 2
    return p


def add_mrp(sigma_left, sigma_right):
    snorm_l = 1 - np.linalg.norm(sigma_left) ** 2
    snorm_r = 1 - np.linalg.norm(sigma_right) ** 2
    new_sigma = snorm_r * sigma_left + snorm_l * sigma_right - 2 * np.cross(sigma_left, sigma_right)
    new_sigma /= (1 + np.linalg.norm(sigma_left) ** 2 * np.linalg.norm(sigma_right) ** 2 - 2 * sigma_left.dot(sigma_right))
    return new_sigma


def dcm_from_mrp(sigma):
    """
    N -> B
    """
    sigma2 = np.linalg.norm(sigma) ** 2
    temp = 1 / (1 + sigma2) ** 2
    c = 8 * skew(sigma).dot(skew(sigma)) - 4 * (1 - sigma2) * skew(sigma)
    c *= temp
    c += np.eye(3)
    return c


# MRP
def get_shadow_set_mrp(sigma):
    sigma_ = -sigma / np.linalg.norm(sigma) ** 2
    return sigma_


def mrp_dot(x):
    sigma_ = x[:3]
    omega_ = x[3:]
    return np.array([*0.25 * Bmatrix_mrp(sigma_).dot(omega_), *np.zeros(3)])


def Bmatrix_mrp(sigma):
    b_matrix = np.zeros((3, 3))
    sigma2 = np.linalg.norm(sigma) ** 2
    for i in range(3):
        b_matrix[i, i] = (1 - sigma2 + 2 * sigma[i] ** 2) * 0.5

    b_matrix[0, 1] = sigma[0] * sigma[1] - sigma[2]
    b_matrix[0, 2] = sigma[0] * sigma[2] + sigma[1]

    b_matrix[1, 0] = sigma[1] * sigma[0] + sigma[2]
    b_matrix[1, 2] = sigma[1] * sigma[2] - sigma[0]

    b_matrix[2, 0] = sigma[2] * sigma[0] - sigma[1]
    b_matrix[2, 1] = sigma[2] * sigma[1] + sigma[0]
    return b_matrix * 2


def matrix_from_vector(d_vector):
    d_ = np.zeros((3, 3))
    d_[0, 0] = d_vector[0]
    d_[1, 1] = d_vector[1]
    d_[2, 2] = d_vector[2]

    d_[0, 1] = d_vector[3]
    d_[0, 2] = d_vector[4]
    d_[1, 0] = d_vector[3]
    d_[2, 0] = d_vector[4]

    d_[1, 2] = d_vector[5]
    d_[2, 1] = d_vector[5]
    return d_


def propagate_rv_by_ang(r_, v_, ang_, mu, ecc):
    f_0 = -np.pi / 2
    ang_ += ang_
    h = np.cross(r_, v_)
    h_norm = np.linalg.norm(h)
    p_ = h_norm ** 2 / mu
    r_p =  p_ / (1 + ecc * np.cos(ang_))
    rot_90 = np.array([[0, -1], [1, 0]])
    new_r = rot_90 @ np.array([np.cos(ang_), np.sin(ang_)]) * r_p
    new_v = np.sqrt(mu / p_) * rot_90 @ np.array([-np.sin(ang_), (ecc + np.cos(ang_))])
    return new_r, new_v
