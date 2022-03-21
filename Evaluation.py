"""
Created by:

@author: Elias Obreque
@Date: 4/19/2021 8:21 PM 
els.obrq@gmail.com

"""
import matplotlib.pyplot as plt

from tools.MonteCarlo import MonteCarlo
from tools.Viewer import *


class Evaluation(object):
    def __init__(self, dynamics, x0, xf, time_options, json_list, control_function, thruster_properties,
                 propellant_properties, type_propellant, folder_name=None):
        self.dynamics = dynamics
        self.x0 = x0
        self.xf = xf
        self.time_options = time_options
        self.json_list = json_list
        self.dynamics.controller_function = control_function
        self.thruster_properties = thruster_properties
        self.propellant_properties = propellant_properties
        self.type_propellant = type_propellant
        self.file_name_1 = "eva_" + type_propellant[:3] + "_Out_data"
        self.file_name_2 = "eva_" + type_propellant[:3] + "_state"
        self.file_name_3 = "eva_" + type_propellant[:3] + "_sigma_Distribution"
        self.file_name_4 = "eva_" + type_propellant[:3] + "_distribution"
        self.file_name_5 = "eva_" + type_propellant[:3] + "_performance"
        self.folder_name = folder_name
        if self.folder_name is None:
            self.folder_name = ""

    def propagate(self, n_case_, n_thrusters_, state_noise=None):
        # # Generation of case (Monte Carlo)
        rN = []
        vN = []
        mN = []

        state_noise_flag = False
        if state_noise is not None:
            state_noise_flag = state_noise[0]
            sdr = state_noise[1]
            sdv = state_noise[2]
            sdm = state_noise[3]
            rN = MonteCarlo(self.x0[0], sdr, n_case_).random_value()
            vN = MonteCarlo(self.x0[1], sdv, n_case_).random_value()
            mN = MonteCarlo(self.x0[2], sdm, n_case_).random_value()

        X_states = []
        THR = []
        IC = []
        EC = []
        TIME = []
        LAND_INDEX = []

        par_force = 1
        i_n = 0
        performance_list = []
        for n_thr in n_thrusters_:
            print("Evaluating with ", n_thr, " number of engine...")
            pulse_thruster = int(n_thr / par_force)

            self.propellant_properties['n_thrusters'] = n_thr
            self.propellant_properties['pulse_thruster'] = pulse_thruster

            self.dynamics.set_engines_properties(self.thruster_properties, self.propellant_properties,
                                                 self.type_propellant)

            if type(self.json_list[str(n_thr)]['Best_individual'][0]) == float:
                for j in range(n_thr):
                    self.dynamics.modify_individual_engine(j, 'alpha',
                                                           self.json_list[str(n_thr)]['Best_individual'][0])
                    self.dynamics.modify_individual_engine(j, 't_burn',
                                                           self.json_list[str(n_thr)]['Best_individual'][1])
            else:
                for j in range(n_thr):
                    self.dynamics.modify_individual_engine(j, 'alpha',
                                                           self.json_list[str(n_thr)]['Best_individual'][0][j])
                    self.dynamics.modify_individual_engine(j, 't_burn',
                                                           self.json_list[str(n_thr)]['Best_individual'][1][j])

            self.dynamics.set_controller_parameters(self.json_list[str(n_thr)]['Best_individual'][2:])
            X_states.append([])
            THR.append([])
            IC.append([])
            EC.append([])
            TIME.append([])
            LAND_INDEX.append([])
            for k in range(n_case_):
                if state_noise_flag:
                    x0_ = [rN[k], vN[k], mN[k]]
                else:
                    x0_ = self.x0

                x_, time_, thrust_, index_control_, end_index_control_, land_i_ = \
                    self.dynamics.run_simulation(x0_, self.xf, self.time_options)

                X_states[i_n].append(x_)
                LAND_INDEX[i_n].append(land_i_)
                THR[i_n].append(thrust_)
                TIME[i_n].append(time_)
                IC[i_n].append(index_control_)
                EC[i_n].append(end_index_control_)

                # Reset thruster
                for thrust in self.dynamics.thrusters:
                    thrust.reset_variables()

            pos_sim = [np.array(X_states[i_n][i])[:, 0] for i in range(n_case_)]
            vel_sim = [np.array(X_states[i_n][i])[:, 1] for i in range(n_case_)]
            mass_sim = [np.array(X_states[i_n][i])[:, 2] for i in range(n_case_)]
            thrust_sim = THR[i_n]

            plot_main_parameters(TIME[i_n], pos_sim, vel_sim, mass_sim, thrust_sim, IC[i_n],
                                 EC[i_n], save=False)
            plot_state_vector(pos_sim, vel_sim, IC[i_n], EC[i_n], folder_name=self.folder_name,
                              file_name=self.file_name_2 + "_" + str(n_thr), save=True)
            performance = plot_distribution(pos_sim, vel_sim, LAND_INDEX[i_n], folder_name=self.folder_name,
                                            file_name=self.file_name_4 + "_" + str(n_thr), save=True)
            performance_list.append(performance)
            close_plot()
            i_n += 1
        plot_performance(performance_list, max(n_thrusters_), folder_name=self.folder_name, file_name=self.file_name_5,
                         save=True)
        plt.show()
        return performance_list


if __name__ == '__main__':
    from Dynamics.Dynamics import Dynamics
    from Thrust.PropellantGrain import propellant_data
    import json

    TUBULAR = 'tubular'
    BATES = 'bates'
    STAR = 'star'
    NEUTRAL = 'neutral'
    PROGRESSIVE = 'progressive'
    REGRESSIVE = 'regressive'

    m0 = 24
    propellant_name = 'TRX-H609'
    selected_propellant = propellant_data[propellant_name]
    propellant_geometry = TUBULAR
    Isp = selected_propellant['Isp']
    den_p = selected_propellant['density']
    ge = 9.807
    c_char = Isp * ge
    g_center_body = -1.62
    r_moon = 1738e3
    mu = 4.9048695e12
    reference_frame = '1D'
    dt = 0.1

    # Initial position for 1D
    r0 = 2000
    v0 = 0

    # Target localization
    rd = 0
    vd = 0

    # Initial and final condition
    x0 = [r0, v0, m0]
    xf = [rd, vd, 0]
    time_options = [0, 1000, dt]

    dynamics = Dynamics(dt, Isp, g_center_body, mu, r_moon, m0, reference_frame, controller='affine_function')

    propellant_properties = {'propellant_name': propellant_name,
                             'n_thrusters': 1,
                             'pulse_thruster': 1,
                             'geometry': None,
                             'propellant_geometry': propellant_geometry,
                             'isp_noise_std': None,
                             'isp_bias_std': None,
                             'isp_dead_time_max': 2}

    engine_diameter_ext = None
    throat_diameter = 1.0  # mm
    height = 10.0  # mm
    file_name = "Thrust/StarGrain7.csv"

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
            return 1
        else:
            return 0

    type_propellant = REGRESSIVE
    name_file = None
    folder_name = "logs/Only_GA_all/"
    if type_propellant == REGRESSIVE:
        folder_name += "regressive/"
        name_file = "reg_Out_data.json"
    elif type_propellant == PROGRESSIVE:
        folder_name += "progressive/"
        name_file = "pro_Out_data.json"
    elif type_propellant == NEUTRAL:
        folder_name += "neutral/"
        name_file = "neu_Out_data.json"
    else:
        print("Select a correct type of propellant grain cross section")

    folder_name += "2022-02-20T14-41-53/"
    f = open(folder_name + name_file)
    data = json.load(f)
    json_list = data

    thruster_properties = {'throat_diameter': 2,
                           'engine_diameter_ext': engine_diameter_ext,
                           'height': height,
                           'performance': {'alpha': 0.0,
                                           't_burn': 0.0},
                           'load_thrust_profile': False,
                           'file_name': file_name,
                           'dead_time': 0.2,
                           'lag_coef': 0.5}

    percentage_variation = 3
    upper_isp = Isp * (1.0 + percentage_variation / 100.0)
    propellant_properties['isp_noise_std'] = (upper_isp - Isp) / 3

    percentage_variation = 10
    upper_isp = Isp * (1.0 + percentage_variation / 100.0)
    propellant_properties['isp_bias_std'] = (upper_isp - Isp) / 3

    n_case = 60
    n_thrusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    evaluation = Evaluation(dynamics, x0, xf, time_options, json_list, control_function, thruster_properties,
                            propellant_properties,
                            type_propellant, folder_name=folder_name[5:])
    eva_performance = evaluation.propagate(n_case, n_thrusters, state_noise=[True, 50.0, 5.0, 0.0])
    json_perf = {'mean_pos': np.array(eva_performance)[:, 0].tolist(),
                 'mean_vel': np.array(eva_performance)[:, 1].tolist(),
                 'std_pos': np.array(eva_performance)[:, 2].tolist(),
                 'std_vel': np.array(eva_performance)[:, 3].tolist()}

    import codecs
    with codecs.open(folder_name + "eva_"+name_file[0:3]+"_performance_data" + ".json", 'w') as file:
        json.dump(json_perf, file)
    print('end')
