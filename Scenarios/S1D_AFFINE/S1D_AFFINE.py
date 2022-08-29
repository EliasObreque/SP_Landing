"""
Created by:

@author: Elias Obreque
@Date: 6/5/2021 11:50 PM 
els.obrq@gmail.com

"""
import time

from datetime import datetime
from tools.GeneticAlgorithm import GeneticAlgorithm
from tools.ext_requirements import mass_req
from Dynamics.Dynamics import Dynamics
from thrust.propellant.propellantGrain import propellant_data
from tools.Viewer import *
from Evaluation import Evaluation
from tools.ext_requirements import save_data

if os.path.isdir("./logs/") is False:
    os.mkdir("./logs/")

# TUBULAR = 'tubular'
# BATES = 'bates'
# STAR = 'star'

NEUTRAL = 'neutral'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'

now = datetime.now()
now = now.strftime("%Y-%m-%dT%H-%M-%S")
reference_frame = '1D'


def s1d_affine(propellant_geometry, type_problem, r0_, v0_, std_alt_, std_vel_, n_case, n_thrusters_,
               full_engine=True, save_plot=True):
    # -----------------------------------------------------------------------------------------------------#
    # Data Mars lander (12U (24 kg), 27U (54 kg))
    m0 = 24
    propellant_name = 'TRX-H609'
    selected_propellant = propellant_data[propellant_name]
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
    r0 = r0_
    v0 = v0_

    # Target localization
    rd = 0
    vd = 0

    # -----------------------------------------------------------------------------------------------------#
    # Initial requirements for 1D
    print('--------------------------------------------------------------------------')
    print('1D requirements')
    dv_req = np.sqrt(2 * r0 * np.abs(g_center_body))
    print('Accumulated velocity[m/s]: ', dv_req)
    mp, m1 = mass_req(dv_req, c_char, den_p, m0)

    # -----------------------------------------------------------------------------------------------------#
    # Simulation time
    dt = 0.1
    simulation_time = 1000
    # -----------------------------------------------------------------------------------------------------#
    # System Propulsion properties
    t_burn_min = 1  # s
    t_burn_max = 20  # s
    n_thruster = 10
    par_force = 1  # Engines working simultaneously

    pulse_thruster = int(n_thruster / par_force)

    total_alpha_min = - g_center_body * m0 / c_char

    # for 1D
    total_alpha_max = mp / t_burn_min
    print('Mass flow rate (1D): (min, max) [kg/s]', total_alpha_min, total_alpha_max)

    max_fuel_mass = 1.05 * mp  # Factor: 1.05

    print('Required engine number for four conditions of mass flow and time burn:'
          ' (min-min, min-max, max-min, max-max) [-]',
          np.ceil(max_fuel_mass / total_alpha_min / t_burn_min),
          np.ceil(max_fuel_mass / total_alpha_min / t_burn_max),
          np.ceil(max_fuel_mass / total_alpha_max / t_burn_min),
          np.ceil(max_fuel_mass / total_alpha_max / t_burn_max))

    T_min = total_alpha_min * c_char
    T_max = total_alpha_max * c_char
    print('Max thrust: (min, max) [N]', T_min, T_max)
    print('--------------------------------------------------------------------------')

    # -----------------------------------------------------------------------------------------------------#
    # Create dynamics object for 1D to calculate optimal mass_flow with ideal constant thrust
    dynamics = Dynamics(dt, Isp, g_center_body, mu, r_moon, m0, reference_frame, controller='basic_hamilton')
    # -----------------------------------------------------------------------------------------------------#
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
    file_name = "thrust/StarGrain7.csv"

    # Optimal solution with GA for constant thrust and multi-engines array
    dynamics.controller_type = 'affine_function'

    type_propellant = propellant_geometry
    if full_engine:
        n_thrusters = list(range(1, n_thrusters_ + 1))
    else:
        n_thrusters = [n_thrusters_]

    # initial condition
    x0 = [r0, v0, m0]
    time_options = [0.0, simulation_time, 0.1]
    xf = [rd, vd, 0]

    print("Initial condition: ", str(x0))
    print("N° case in training: ", n_case)
    print("Type of propellant: ", type_propellant)
    print("Type of problem: ", type_problem)
    state_noise = None

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
    elif type_problem == 'isp_bias-noise':
        percentage_variation = 3
        upper_isp = Isp * (1.0 + percentage_variation / 100.0)
        propellant_properties['isp_noise_std'] = (upper_isp - Isp) / 3
        percentage_variation = 10
        upper_isp = Isp * (1.0 + percentage_variation / 100.0)
        propellant_properties['isp_bias_std'] = (upper_isp - Isp) / 3
        propellant_properties['isp_dead_time_max'] = 2
    elif type_problem == 'state_noise':
        std_alt = std_alt_
        state_noise = [True, std_alt_, std_vel_]
    elif type_problem == 'all':
        percentage_variation = 3
        upper_isp = Isp * (1.0 + percentage_variation / 100.0)
        propellant_properties['isp_noise_std'] = (upper_isp - Isp) / 3
        percentage_variation = 10
        upper_isp = Isp * (1.0 + percentage_variation / 100.0)
        propellant_properties['isp_bias_std'] = (upper_isp - Isp) / 3
        propellant_properties['isp_dead_time_max'] = 2
        std_alt = std_alt_
        std_vel = std_vel_
        state_noise = [True, std_alt, std_vel]
    else:
        n_case = 1

    # Calculate optimal alpha (m_dot) for a given t_burn and constant ideal thrust
    t_burn = 0.5 * (t_burn_min + t_burn_max)
    total_alpha_max = 0.9 * m0 / t_burn
    optimal_alpha = dynamics.basic_hamilton_calc.calc_simple_optimal_parameters(x0[0], total_alpha_min,
                                                                                total_alpha_max,
                                                                                t_burn)

    thruster_properties = {'throat_diameter': 2,
                           'engine_diameter_ext': engine_diameter_ext,
                           'height': height,
                           'performance': {'alpha': optimal_alpha,
                                           't_burn': t_burn},
                           'load_thrust_profile': False,
                           'file_name': file_name,
                           'dead_time': 0.2,
                           'lag_coef': 0.5}

    dynamics.set_engines_properties(thruster_properties, propellant_properties)

    poly = [g_center_body/2, v0, r0]
    root = np.roots(poly)
    t_free = max(root)

    def sp_cost_function(ga_x_states, thr, time_ser, ga_land_index, Ah, Bh):
        error_pos = ga_x_states[ga_land_index][0] - xf[0]
        error_vel = ga_x_states[ga_land_index][1] - xf[1]
        if max(np.array(ga_x_states)[:, 1]) > 0:
            error_vel *= 10
        if min(np.array(ga_x_states)[:, 0]) < 0:
            error_pos *= 100
        rate_time = max(time_ser) / t_free
        return Ah * error_pos ** 2 + Bh * error_vel ** 2# + rate_time * 10

    json_list = {}
    file_name_1 = type_propellant[:3] + "_Out_data"
    file_name_2 = type_propellant[:3] + "_state"
    file_name_3 = type_propellant[:3] + "_sigma_Distribution"
    file_name_4 = type_propellant[:3] + "_distribution"
    file_name_5 = type_propellant[:3] + "_performance"
    json_list['N_case'] = n_case
    performance_list = []
    folder_name = "Only_GA_" + str(type_problem) + "/" + type_propellant + "/" + now + "/"

    for n_thr in n_thrusters:
        print('N thrust: ', n_thr)
        json_list[str(n_thr)] = {}
        pulse_thruster = int(n_thr / par_force)

        propellant_properties['n_thrusters'] = n_thr
        propellant_properties['pulse_thruster'] = pulse_thruster

        alpha_min = total_alpha_min/pulse_thruster
        alpha_max = optimal_alpha * 2 / pulse_thruster
        # 300 - 30

        # if type_propellant != NEUTRAL:
        #     t_burn_max = (space_max / np.sqrt(n_thr) / thickness_case_factor) / 30 * 8.0

        ga = GeneticAlgorithm(max_generation=300, n_individuals=50,
                              ranges_variable=[['float', alpha_min, alpha_max, pulse_thruster],
                                               ['float', 0.0, t_burn_max, pulse_thruster], ['str', type_propellant],
                                               ['float_iter', 0.0, 1.0, pulse_thruster],
                                               ['float_iter', 0.0, x0[0] / np.sqrt(2 * np.abs(g_center_body) * x0[0]),
                                                pulse_thruster]],
                              mutation_probability=0.25)

        start_time = time.time()
        best_states, best_time_data, best_Tf, best_individuals, index_control, end_index_control, land_index = ga.optimize(
            cost_function=sp_cost_function, n_case=n_case, restriction_function=[dynamics, x0, xf, time_options,
                                                                                 propellant_properties,
                                                                                 thruster_properties],
            alt_noise=state_noise)

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
        # plot_sigma_distribution(best_pos, best_vel, land_index, folder_name, file_name_3, lim_std3sigma, save=save_plot)
        performance = plot_distribution(best_pos, best_vel, land_index, folder_name, file_name_4 + "_" + str(n_thr),
                                        save=save_plot)
        performance_list.append(performance)
        json_list[str(n_thr)]['performance'] = {'mean_pos': performance[0],
                                                'mean_vel': performance[1],
                                                'std_pos': performance[2],
                                                'std_vel': performance[3]}

        plot_main_parameters(best_time_data, best_pos, best_vel, best_mass, best_thrust, index_control,
                             end_index_control, save=save_plot, folder_name=folder_name, file_name=file_name_1)
        plot_state_vector(best_pos, best_vel, index_control, end_index_control, save=save_plot,
                          folder_name=folder_name, file_name=file_name_2 + "_" + str(n_thr))

        close_plot()

    save_data(json_list, folder_name, file_name_1)
    if len(n_thrusters) != 1:
        plot_performance(performance_list, max(n_thrusters), save=save_plot, folder_name=folder_name,
                         file_name=file_name_5)
        print('Performance plot saved')

    # %%
    #   Evaluation
    print("--------------------------------------------------------------------------------------------------------")
    print("Start Evaluation process...")

    def control_function(control_par, current_state, type_control='affine'):
        a = control_par[0]
        b = control_par[1]
        current_alt = current_state[0]
        current_vel = current_state[1]
        f = 0
        if type_control == 'affine':
            f = a * current_alt + b * current_vel
        elif type_control == 'pol2':
            f = a * current_alt - b * current_vel ** 2
        elif type_control == 'pol3':
            c = control_par[2]
            f = a * current_alt - b * current_vel ** 2 + c * current_vel ** 3
        if f <= 0:
            return 1, f
        else:
            return 0, f

    # percentage_variation = 3
    # upper_isp = Isp * (1.0 + percentage_variation / 100.0)
    # propellant_properties['isp_noise_std'] = (upper_isp - Isp) / 3
    #
    # percentage_variation = 10
    # upper_isp = Isp * (1.0 + percentage_variation / 100.0)
    # propellant_properties['isp_bias_std'] = (upper_isp - Isp) / 3
    n_case_eval = 1
    # propellant_properties['isp_dead_time_max'] = 2

    evaluation = Evaluation(dynamics, x0, xf, time_options, json_list, control_function, thruster_properties,
                            propellant_properties,
                            type_propellant, folder_name)
    eva_performance = evaluation.propagate(n_case_eval, n_thrusters, state_noise=[False, std_alt_, std_vel_, 0.0])

    json_perf = {'mean_pos': np.array(eva_performance)[:, 0].tolist(),
                 'mean_vel': np.array(eva_performance)[:, 1].tolist(),
                 'std_pos': np.array(eva_performance)[:, 2].tolist(),
                 'std_vel': np.array(eva_performance)[:, 3].tolist()}
    save_data(json_perf, folder_name, "eva_" + type_propellant[:3] + "_performance_data")
    print("Finished")


if __name__ == '__main__':
    r0_, v0_, std_alt_, std_vel_, n_case_train, n_thrusters_ = 2000.0, 0.0, 50.0, 5.0, 30, 10

    # Problem: "isp_noise"-"isp_bias"-"normal"-"isp_bias-noise"-"alt_noise"-"all" - "no_noise"

    type_problem = "all"
    propellant_geometry = NEUTRAL
    s1d_affine(propellant_geometry, type_problem, r0_, v0_, std_alt_, std_vel_, n_case_train, n_thrusters_)


