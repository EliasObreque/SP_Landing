"""
Created: 9/14/2020
Author: Elias Obreque Sepulveda
email: els.obrq@gmail.com

array_propellant_names = ['JPL_540A', 'ANP-2639AF', 'CDT(80)', 'TRX-H609', 'KNSU']

"""
import time
import sys
import codecs
import json
from datetime import datetime
from tools.GeneticAlgorithm import GeneticAlgorithm
from tools.ext_requirements import velocity_req, mass_req
from Dynamics.Dynamics import Dynamics
from Thrust.PropellantGrain import propellant_data
from tools.Viewer import *
from Evaluation import Evaluation

if os.path.isdir("./logs/") is False:
    os.mkdir("./logs/")

CONSTANT  = 'constant'
TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'

ONE_D = '1D'
POLAR = 'polar'
now = datetime.now()
now = now.strftime("%Y-%m-%dT%H-%M-%S")
reference_frame = ONE_D
# -----------------------------------------------------------------------------------------------------#
# Data Mars lander (12U (24 kg), 27U (54 kg))
m0 = 24
propellant_name = 'CDT(80)'
selected_propellant = propellant_data[propellant_name]
propellant_geometry = TUBULAR
Isp = selected_propellant['Isp']
den_p = selected_propellant['density']
ge = 9.807
c_char = Isp * ge

# Available space for engine (square)
space_max = 180  # mm
thickness_case_factor = 1.2
aux_dimension = 150  # mm
d_int = 2  # mm


# -----------------------------------------------------------------------------------------------------#
# Center body data
# Moon
g_center_body = -1.62
r_moon = 1738e3
mu = 4.9048695e12

# -----------------------------------------------------------------------------------------------------#
# Initial position for 1D
r0 = 2000
v0 = 0

# Target localization
rd = 0
vd = 0

# -----------------------------------------------------------------------------------------------------#
# Initial position for Polar coordinate
rrp = 2000e3
rra = 68000e3
ra = 0.5 * (rra + rrp)
# Orbital velocity
vp = np.sqrt(mu * (2 / rrp - 1 / ra))
va = np.sqrt(mu * (2 / rra - 1 / ra))
print('Perilune velocity [m/s]: ', vp)
print('Apolune velocity [m/s]: ', va)
moon_period = 2 * np.pi * np.sqrt(ra ** 3 / mu)

p_r0 = rrp
p_v0 = 0
p_theta0 = 0
p_omega0 = vp / rrp
p_m0 = m0

# Target localization
p_rf = r_moon
p_vf = 0
p_thetaf = 0
p_omegaf = 0
p_mf = m0

# -----------------------------------------------------------------------------------------------------#
# Initial requirements for 1D
print('--------------------------------------------------------------------------')
print('1D requirements')
dv_req = np.sqrt(2 * r0 * np.abs(g_center_body))
print('Accumulated velocity[m/s]: ', dv_req)
mp, m1 = mass_req(dv_req, c_char, den_p, m0)

# Initial requirements for polar
print('\nPolar requirements')
dv_req_p, dv_req_a = velocity_req(vp, va, r_moon, mu, rrp, rra)
p_mp, p_m1 = mass_req(dv_req_p, c_char, den_p, m0)
print('--------------------------------------------------------------------------')

# -----------------------------------------------------------------------------------------------------#
# Simulation time
dt = 0.1
simulation_time = moon_period
# -----------------------------------------------------------------------------------------------------#
# System Propulsion properties
t_burn_min = 3  # s
t_burn_max = 40  # s
n_thruster = 10
par_force = 1  # Engines working simultaneously

pulse_thruster = int(n_thruster / par_force)

total_alpha_min = - g_center_body * m0 / c_char

# for 1D
total_alpha_max = mp / t_burn_min
print('Mass flow rate (1D): (min, max) [kg/s]', total_alpha_min, total_alpha_max)

# for Polar
total_alpha_max_p = p_mp / t_burn_min
print('Mass flow rate (Polar): (min, max) [kg/s]', total_alpha_min, total_alpha_max_p)

max_fuel_mass = 1.05 * mp  # Factor: 1.05

print('Required engines: (min-min, min-max, max-min, max-max) [-]',
      max_fuel_mass / total_alpha_min / t_burn_min,
      max_fuel_mass / total_alpha_min / t_burn_max,
      max_fuel_mass / total_alpha_max / t_burn_min,
      max_fuel_mass / total_alpha_max / t_burn_max)

print('Required engines: (min-min, min-max, max-min, max-max) [-]',
      max_fuel_mass / total_alpha_min / t_burn_min,
      max_fuel_mass / total_alpha_min / t_burn_max,
      max_fuel_mass / total_alpha_max_p / t_burn_min,
      max_fuel_mass / total_alpha_max_p / t_burn_max)

T_min = total_alpha_min * c_char
T_max = total_alpha_max * c_char
print('Max thrust: (min, max) [N]', T_min, T_max)
print('--------------------------------------------------------------------------')


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


# -----------------------------------------------------------------------------------------------------#
# Create dynamics object for 1D and Polar
dynamics = Dynamics(dt, Isp, g_center_body, mu, r_moon, m0, reference_frame, controller='basic_hamilton')
# -----------------------------------------------------------------------------------------------------#
# Simple example solution with constant one engine for 1D
dynamics.calc_limits_by_single_hamiltonian(t_burn_min, t_burn_max, total_alpha_min, total_alpha_max, plot_data=True)

# Calculate optimal alpha (m_dot) for a given t_burn
t_burn = 0.5 * (t_burn_min + t_burn_max)
total_alpha_max = 0.9 * m0 / t_burn
optimal_alpha = dynamics.basic_hamilton_calc.calc_simple_optimal_parameters(r0, total_alpha_min, total_alpha_max,
                                                                            t_burn)
# Define propellant properties to create a Thruster object with the optimal alpha
propellant_properties = {'propellant_name': propellant_name,
                         'n_thrusters': 1,
                         'pulse_thruster': 1,
                         'geometry': None,
                         'propellant_geometry': propellant_geometry,
                         'isp_noise_std': None,
                         'isp_bias_std': None,
                         'isp_dead_time_max': None}

engine_diameter_ext = None
throat_diameter = 1.0  # mm
height = 10.0  # mm
file_name = "Thrust/StarGrain7.csv"
thruster_properties = {'throat_diameter': 2,
                       'engine_diameter_ext': engine_diameter_ext,
                       'height': height,
                       'performance': {'alpha': optimal_alpha,
                                       't_burn': t_burn},
                       'load_thrust_profile': False,
                       'file_name': file_name,
                       'delay': None}

dynamics.set_engines_properties(thruster_properties, propellant_properties)

# Initial and final condition
x0 = [r0, v0, m0]
xf = [0.0, 0.0, m1]
time_options = [0, simulation_time, dt]

x_states, time_series, thr, _, _, _ = dynamics.run_simulation(x0, xf, time_options)
dynamics.basic_hamilton_calc.print_simulation_data(x_states, mp, m0, r0)
dynamics.basic_hamilton_calc.plot_1d_simulation(x_states, time_series, thr)

# -----------------------------------------------------------------------------------------------------#
# Optimal solution with GA for constant thrust and multi-engines array
t_burn_min, t_burn_max   = 1, 30
dynamics.controller_type = 'ga_wo_hamilton'

r0              = 2000
type_problem    = "isp_dead_time"
type_propellant = CONSTANT
n_case          = 30  # Case number
n_thrusters      = [1, 2, 3]

if len(sys.argv) > 1:
    print(list(sys.argv))
    if len(list(sys.argv)) == 2:
        if sys.argv[1] == 'help':
            print('<altitude> ', '<type_problem> ', '<type_propellant>')
            print('altitude: [m]')
            print('type_problem: "isp_noise"-"isp_bias"-"isp_dead_time"-"isp_bias-noise"-"alt_noise"-"all" - "no_noise"')
            print('type_propellant: "constant" - "progressive" - "regressive"')
            print('---')
            sys.exit()

    r0 = float(sys.argv[1])  # altitude [m]
    type_problem    = sys.argv[2]   # Problem: "isp_noise"-"isp_bias"-"normal"-"isp_bias-noise"-"alt_noise"-"all" - "no_noise"
    type_propellant = sys.argv[3]

# initial condition
x0 = [r0, v0, m0]
time_options = [0.0, simulation_time, 0.1]

print("Initial condition: ", str(x0))
print("N_case: ", n_case)
print("type_propellant: ", type_propellant)
print("type_problem: ", type_problem)
alt_noise = None

# +-10% and multi-engines array
# gauss_factor = 1 for 68.3%, = 2 for 95.45%, = 3 for 99.74%
propellant_properties['isp_noise_std'] = None
propellant_properties['isp_bias_std'] = None
propellant_properties['isp_dead_time_max'] = 0
if type_problem == 'isp_noise':
    percentage_variation = 1
    upper_isp = Isp * (1.0 + percentage_variation / 100.0)
    propellant_properties['isp_noise_std'] = (upper_isp - Isp) / 3
elif type_problem == 'isp_bias':
    percentage_variation = 10
    upper_isp = Isp * (1.0 + percentage_variation / 100.0)
    propellant_properties['isp_bias_std'] = (upper_isp - Isp) / 3
elif type_problem == 'isp_dead_time':
    propellant_properties['isp_dead_time_max'] = 1
elif type_problem == 'isp_bias-noise':
    percentage_variation = 1
    upper_isp = Isp * (1.0 + percentage_variation / 100.0)
    propellant_properties['isp_noise_std'] = (upper_isp - Isp) / 3
    percentage_variation = 10
    upper_isp = Isp * (1.0 + percentage_variation / 100.0)
    propellant_properties['isp_bias_std'] = (upper_isp - Isp) / 3
    propellant_properties['isp_dead_time_max'] = 2
elif type_problem == 'alt_noise':
    std_alt = 50
    alt_noise = [True, std_alt]

# Calculate optimal alpha (m_dot) for a given t_burn and constant ideal thrust
t_burn = 0.5 * (t_burn_min + t_burn_max)
total_alpha_max = 0.9 * m0 / t_burn
optimal_alpha = dynamics.basic_hamilton_calc.calc_simple_optimal_parameters(x0[0], total_alpha_min,
                                                                            total_alpha_max,
                                                                            t_burn)


def sp_cost_function(ga_x_states, thr, time_ser, ga_land_index, Ah, Bh):
    error_pos = ga_x_states[ga_land_index][0] - xf[0]
    error_vel = ga_x_states[ga_land_index][1] - xf[1]
    if max(np.array(ga_x_states)[:, 1]) > 0:
        error_vel *= 10
    if max(np.array(ga_x_states)[:, 0]) < 0:
        error_pos *= 100
    # return (Ah * error_pos ** 2 + Bh * error_vel ** 2)/time_ser[-1] #+ (time_ser[-1] + error_pos ** 2 + error_vel ** 2)/time_ser[-1]
    return Ah * error_pos ** 2 + Bh * error_vel ** 2 + 10 * (ga_x_states[0][2] / ga_x_states[-1][2]) ** 2


json_list = {}
file_name_1 = "Out_data"
file_name_2 = "State_variable"
file_name_3 = "Sigma_Distribution"
file_name_4 = "Normal_Distribution"
file_name_5 = "Performance_by_motor"

json_list['N_case'] = n_case
thruster_properties['delay'] = 0.3
performance_list = []
for n_thr in n_thrusters:
    print('N thrust: ', n_thr)
    json_list[str(n_thr)] = {}
    pulse_thruster = int(n_thr / par_force)

    propellant_properties['n_thrusters'] = n_thr
    propellant_properties['pulse_thruster'] = pulse_thruster

    alpha_min = total_alpha_min/pulse_thruster
    alpha_max = optimal_alpha * 2 / pulse_thruster
    # 300 - 30

    # if type_propellant != CONSTANT:
    #     t_burn_max = (space_max / np.sqrt(n_thr) / thickness_case_factor) / 30 * 8.0

    ga = GeneticAlgorithm(max_generation=250, n_individuals=40,
                          ranges_variable=[['float', alpha_min, alpha_max, pulse_thruster],
                                           ['float', 0.0, t_burn_max, pulse_thruster], ['str', type_propellant],
                                           ['float_iter', 0.0, 1.0, pulse_thruster],
                                           ['float_iter', 0.0, x0[0] / np.sqrt(2 * np.abs(g_center_body) * x0[0]),
                                            pulse_thruster]],
                          mutation_probability=0.2)

    start_time = time.time()
    best_states, best_time_data, best_Tf, best_individuals, index_control, end_index_control, land_index = ga.optimize(
        cost_function=sp_cost_function, n_case=n_case, restriction_function=[dynamics, x0, xf, time_options,
                                                                             propellant_properties,
                                                                             thruster_properties], alt_noise=alt_noise)
    finish_time = time.time()
    print('Time to optimize: ', finish_time - start_time, '[s]')

    best_pos    = [best_states[i][:, 0] for i in range(n_case)]
    best_vel    = [best_states[i][:, 1] for i in range(n_case)]
    best_mass   = [best_states[i][:, 2] for i in range(n_case)]
    best_thrust = best_Tf

    for k in range(n_case):
        json_list[str(n_thr)]['Case' + str(k)] = {}
        df = {'Time[s]': best_time_data[k].tolist(), 'Pos[m]': best_pos[k].tolist(),
              'V[m/s]': best_vel[k].tolist(), 'mass[kg]': best_mass[k].tolist(),
              'T[N]': best_thrust[k].tolist()}
        df_cost = {'Cost_function[-]': np.array(ga.historical_cost)[:, k].tolist()}

        json_list[str(n_thr)]['Case' + str(k)]['response'] = df
        json_list[str(n_thr)]['Case' + str(k)]['cost'] = df_cost

    folder_name = "Only_GA_" + str(type_problem) + "/" + type_propellant + "/" + str(int(x0[0])) + "m_" + now + "/"\
                  + "n_thr-" + str(n_thr) + "/"

    json_list[str(n_thr)]['Best_individual'] = [best_individuals[0], best_individuals[1], best_individuals[3],
                                                best_individuals[4]]
    print('Best individual for ', n_thr, 'engines')
    print('m_dot: ', np.round(best_individuals[0], 5), '[kg/s]')
    print('t_burn: ', np.round(best_individuals[1], 5), '[s]')
    print('a: ', np.round(best_individuals[3], 5), '[-]')
    print('b: ', np.round(best_individuals[4], 5), '[-]')
    print('c: ', 0, '[-]')
    print('--------------------------------------------------------')

    lim_std3sigma = [1, 3]  # [m, m/S]
    plot_sigma_distribution(best_pos, best_vel, land_index, folder_name, file_name_3, lim_std3sigma, save=True)
    performance = plot_gauss_distribution(best_pos, best_vel, land_index, folder_name, file_name_4, save=True)
    performance_list.append(performance)
    json_list[str(n_thr)]['performance'] = {'mean_pos': performance[0],
                                            'mean_vel': performance[1],
                                            'std_pos': performance[2],
                                            'std_vel': performance[3]}

    plot_main_parameters(best_time_data, best_pos, best_vel, best_mass, best_thrust, index_control,
                         end_index_control, save=True, folder_name=folder_name, file_name=file_name_1)
    plot_state_vector(best_pos, best_vel, index_control, end_index_control, save=True,
                      folder_name=folder_name, file_name=file_name_2)

    close_plot()

file_name_1 = "Out_data"
folder_name = "Only_GA_" + str(type_problem) + "/" + type_propellant + "/" + str(int(x0[0])) + "m_" + now + "/"
save_data(json_list, folder_name, file_name_1)

plot_performance(performance_list, max(n_thrusters), save=True, folder_name=folder_name, file_name=file_name_5)
print('Performance plot saved')

#   Evaluation

def control_function(control_par, current_state):
    a = control_par[0]
    b = control_par[1]
    current_alt = current_state[0]
    current_vel = current_state[1]
    f = a * current_alt + b * current_vel
    if f <= 0:
        return 1
    else:
        return 0


n_case = 60
evaluation = Evaluation(dynamics, x0, xf, time_options, json_list, control_function, thruster_properties,
                        propellant_properties,
                        type_propellant)
evaluation.propagate(n_case, n_thrusters, state_noise=[True, 100.0, .0, 0.0])

print("Finished")
