"""
Created: 7/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from Thrust.Thruster import Thruster


class GeneticAlgorithm(object):
    def __init__(self, step_w, g_planet, init_state, max_generation, n_individuals, range_variables):
        self.init_state = init_state
        self.g_planet = g_planet
        self.step_width = step_w
        self.Ah = 0.01
        self.Bh = 10.0
        self.Ch = 0.1
        self.selection_method = 'rank'
        self.current_cost = []
        self.max_generation = max_generation
        self.n_individuals = n_individuals
        self.range_variables = range_variables
        self.n_variables = len(range_variables)
        self.ind_selected = []
        self.weight_crossover = 0.2
        self.n_thrust = 0
        self.prob_mut = 0.1
        self.thrust_comp = None
        self.population = []
        self.historical_cost = []
        self.create_first_population()
        self.dynamics_update = None
        self.c_char = 200
        self.betas = []
        self.save_impact_velocity = None

    def create_first_population(self):
        for k in range(self.n_individuals):
            individual = []
            for i in range(self.n_variables):
                if self.range_variables[i][0] == 'float':
                    temp = np.random.uniform(self.range_variables[i][1], self.range_variables[i][2])
                    individual.append(temp)
                elif self.range_variables[i][0] == 'int':
                    if len(self.range_variables[i]) == 3:
                        temp = np.random.randint(self.range_variables[i][1], self.range_variables[i][2])
                    else:
                        temp = self.range_variables[i][1]
                        self.n_thrust = temp
                    individual.append(temp)
                elif self.range_variables[i][0] == 'str':
                    type_pro = np.random.randint(1, len(self.range_variables[i]))
                    temp = self.range_variables[i][type_pro]
                    individual.append(temp)
                elif self.range_variables[i][0] == 'float_iter':
                    temp = []
                    for m in range(individual[2]):
                        temp.append(np.random.uniform(self.range_variables[i][1], self.range_variables[i][2]))
                        individual.append(temp)
            self.population.append(individual)
        return

    def optimize(self, dynamics_update, c_char):
        print('Running...')
        self.c_char = c_char
        self.dynamics_update = dynamics_update
        ge = 1
        Hf, Vf, Mf, Tf = self.ga_evaluate()
        print('Generation: ', ge, ', Cost: ', min(self.current_cost))
        best_index       = self.current_cost.index(min(self.current_cost))
        best_individuals = self.population[best_index]
        best_cost        = min(self.current_cost)
        best_Hf, best_Vf, best_Mf, best_Tf = Hf[best_index], Vf[best_index], Mf[best_index], Tf[best_index]
        self.historical_cost.append(min(self.current_cost))
        percent = 0.8
        n_indiv_by_selec = round(self.n_individuals * percent)
        rest_ind = n_indiv_by_selec % 2
        n_indiv_by_selec -= rest_ind
        while ge < self.max_generation:  # or max(self.current_cost) <= 0.5:
            next_population = []
            crossover_arith = True
            if self.n_individuals - n_indiv_by_selec != 0:
                descent = self.ga_selection(self.n_individuals - n_indiv_by_selec, direct_method=True)
                next_population += descent
            for i in range(int(n_indiv_by_selec * 0.5)):
                index_parents = self.ga_selection(2)
                if crossover_arith:
                    descent1, descent2 = self.ga_crossover_arithmetic(index_parents[0], index_parents[1])
                    crossover_arith = False
                else:
                    descent1, descent2 = self.ga_crossover_coding(index_parents[0], index_parents[1])
                    crossover_arith = True
                new_generation1 = self.ga_mutation(descent1, True)
                new_generation2 = self.ga_mutation(descent2, True)
                next_population.append(new_generation1)
                next_population.append(new_generation2)
            del self.population
            self.population = next_population
            Hf, Vf, Mf, Tf = self.ga_evaluate()
            ge += 1
            if ge > self.max_generation:
                self.max_generation = ge
            print('Generation: ', ge, ', Cost: ', min(self.current_cost))
            self.historical_cost.append(min(self.current_cost))
            if min(self.current_cost) < best_cost:
                best_index = self.current_cost.index(min(self.current_cost))
                best_Hf, best_Vf, best_Mf, best_Tf = Hf[best_index], Vf[best_index], \
                                                     Mf[best_index], Tf[best_index]
                best_cost = min(self.current_cost)
                best_individuals = self.population[best_index]

        print(best_cost)
        print(best_individuals)
        self.plot_cost()
        best_pos = np.array(best_Hf)
        best_vel = np.array(best_Vf)
        best_mass = np.array(best_Mf)
        best_thrust = best_Tf

        self.plot_best(best_pos, best_vel, best_mass, best_thrust, best_individuals)
        self.plot_state_vector(best_pos, best_vel, best_individuals)
        # self.print_report(best_pos, best_vel, best_mass, best_thrust, best_individuals)
        return

    def plot_cost(self):
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Optimization function')
        plt.plot(np.arange(1, self.max_generation + 1), np.array(self.historical_cost), label='Min')
        plt.grid()
        plt.legend()
        plt.show()

    def ga_evaluate(self):
        self.current_cost = []
        POS  = []
        VEL  = []
        MASS = []
        THR  = []
        for indv in range(self.n_individuals):
            k  = 0
            x1 = [self.init_state[0][0]]
            x2 = [self.init_state[1][0]]
            x3 = [self.init_state[2]]
            thr = []
            comp_thrust = []
            end_condition = False
            for j in range(self.population[indv][2]):
                comp_thrust.append(Thruster(self.step_width, max_burn_time=self.population[indv][1],
                                            nominal_thrust=self.population[indv][0] * self.c_char,
                                            type_propellant=self.population[indv][3]))
                comp_thrust[j].set_lag_coef(0.15)

            while end_condition is False:
                total_thr = 0
                for j in range(self.population[indv][2]):
                    current_beta = self.get_beta(self.population[indv][4][j], x1[k])
                    comp_thrust[j].set_beta(current_beta)
                    comp_thrust[j].calc_thrust_mag(self.step_width * 1000)
                    total_thr += comp_thrust[j].current_mag_thrust_c
                thr.append(total_thr)
                next_state = self.rungeonestep(thr[k], x1[k], x2[k], x3[k])

                x1.append(next_state[0])
                x2.append(next_state[1])
                x3.append(next_state[2])
                k += 1
                if x1[k] < 0 and thr[k - 1] == 0.0:
                    end_condition = True
                    x1.pop(k)
                    x2.pop(k)
                    x3.pop(k)

            jcost = self.calc_cost_function(x1, x2, x3, thr, indv)
            self.current_cost.append(jcost)
            self.save_impact_velocity = None
            POS.append(x1)
            VEL.append(x2)
            MASS.append(x3)
            THR.append(thr)
        return POS, VEL, MASS, THR

    def get_beta(self, ignition_alt, current_alt):
        if current_alt <= ignition_alt:
            return 1
        else:
            return 0

    def ga_selection(self, n, direct_method=False):
        if direct_method is False:
            if self.selection_method == 'roulette':
                probability_selection = np.array(self.current_cost) / np.sum(self.current_cost)
                ind_selected = np.random.choice(a=np.arange(self.n_individuals),
                                                size=n,
                                                p=list(probability_selection),
                                                replace=True)
            elif self.selection_method == "rank":
                ranks = rankdata(1 * np.array(self.current_cost))
                selection_probability = 1 / ranks
                selection_probability = selection_probability / np.sum(selection_probability)
                ind_selected = np.random.choice(a=np.arange(self.n_individuals),
                                                size=n,
                                                p=list(selection_probability),
                                                replace=True)
            else:
                print('Select method')
                ind_selected = []
        else:
            list_order = sorted(self.current_cost)
            value_select = list_order[:n]
            ind_selected = []
            for elem in value_select:
                index_select = np.where(self.current_cost == elem)[0][0]
                ind_selected.append(self.population[int(index_select)])
        return ind_selected

    def ga_crossover_coding(self, parents_1, parents_2):
        children_1 = []
        children_2 = []
        cut_position = np.random.randint(0, self.n_variables)
        children_1[:cut_position] = self.population[parents_1][:cut_position]
        children_1[cut_position:] = self.population[parents_2][cut_position:]
        children_2[:cut_position] = self.population[parents_2][:cut_position]
        children_2[cut_position:] = self.population[parents_1][cut_position:]
        return children_1, children_2

    def ga_crossover_arithmetic(self, parents_1, parents_2):
        children_1 = []
        children_2 = []
        for i in range(self.n_variables):
            if type(self.population[parents_1][i]) == float:
                chs1 = self.population[parents_1][i] * self.weight_crossover \
                       + (1 - self.weight_crossover) * self.population[parents_2][i]
                chs2 = self.population[parents_2][i] * self.weight_crossover \
                       + (1 - self.weight_crossover) * self.population[parents_1][i]
                children_1.append(chs1)
                children_2.append(chs2)
            elif type(self.population[parents_1][i]) == str:
                chs1 = self.population[parents_1][i]
                chs2 = self.population[parents_2][i]
                children_1.append(chs1)
                children_2.append(chs2)
            elif type(self.population[parents_1][i]) == list:
                sub_chs1 = []
                sub_chs2 = []
                for k in range(len(self.population[parents_1][i])):
                    sub_chs1.append(self.population[parents_1][i][k] * self.weight_crossover + \
                                    (1 - self.weight_crossover) * self.population[parents_2][i][k])
                    sub_chs2.append(self.population[parents_2][i][k] * self.weight_crossover + \
                                    (1 - self.weight_crossover) * self.population[parents_1][i][k])
                children_1.append(sub_chs1)
                children_2.append(sub_chs2)
            elif type(self.population[parents_1][i]) == int:
                chs1 = self.population[parents_1][i]
                chs2 = self.population[parents_2][i]
                children_1.append(chs1)
                children_2.append(chs2)
            else:
                print('No type found')
        return children_1, children_2

    def ga_mutation(self, descent, mutate):
        if not mutate:
            return descent
        else:
            len_vec = self.n_variables + self.n_thrust - 2
            mutated_positions = np.random.uniform(low=0, high=1, size=len_vec)
            mutated_positions = mutated_positions < self.prob_mut
            if np.all(~mutated_positions):
                return descent
            else:
                k = 0
                for elem in mutated_positions:
                    if elem and k < 2:
                        descent[k] += 0.5 * descent[k] * np.random.normal(0, 0.2)
                        if descent[k] < self.range_variables[k][1]:
                            descent[k] = self.range_variables[k][1]
                        elif descent[k] > self.range_variables[k][2]:
                            descent[k] = self.range_variables[k][2]
                        else:
                            descent[k] = descent[k]
                        k += 1
                    elif elem and k > 5:
                        descent[4][k - 5] += 0.5 * descent[4][k - 5] * np.random.normal(0, 0.2)
                        if descent[4][k - 5] < self.range_variables[4][2]:
                            descent[4][k - 5] = self.range_variables[4][2]
                        elif descent[4][k - 5] > self.range_variables[4][1]:
                            descent[4][k - 5] = self.range_variables[4][1]
                        else:
                            descent[4][k - 5] = descent[4][k - 5]
                        k += 1
                    else:
                        k += 1
        return descent

    def calc_cost_function(self, pos_, vel_, mass_, thrust, k):
        error_pos = pos_[-1] - self.init_state[0][1]
        if self.save_impact_velocity is not None:
            error_vel = self.save_impact_velocity - self.init_state[1][1]
        else:
            error_vel = vel_[-1] - self.init_state[1][1]
        max_engine = 2
        current_alpha = self.population[k][0]
        current_engine = np.max(thrust) / current_alpha / self.c_char
        return self.Ah * error_pos ** 2 + self.Bh * error_vel ** 2 + (1 - current_engine / max_engine) ** 2

    def dynamics(self, state, t, T):
        x = state[0]
        vx = state[1]
        mass = state[2]
        rhs = np.zeros(3)
        rhs[0] = vx
        rhs[1] = self.g_planet + T / mass
        rhs[2] = -T / self.c_char
        return rhs

    def rungeonestep(self, T, pos, vel, mass):
        dt = self.step_width
        t = 0
        if pos <= 0 and self.save_impact_velocity is None:
            self.save_impact_velocity = vel
            vel = 0
            pos = 0
        x = np.array([pos, vel, mass])
        k1 = self.dynamics(x, t, T)
        xk2 = x + (dt / 2.0) * k1
        k2 = self.dynamics(xk2, (t + dt / 2.0), T)
        xk3 = x + (dt / 2.0) * k2
        k3 = self.dynamics(xk3, (t + dt / 2.0), T)
        xk4 = x + dt * k3
        k4 = self.dynamics(xk4, (t + dt), T)
        next_x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_x

    def calc_force_torque(self):
        self.force_ = 0

    def get_force_i(self):
        return self.force_

    def plot_state_vector(self, best_pos, best_vel, best_individuals):
        plt.figure()
        ini_best_individuals = [np.array(np.abs(best_pos - i)).argmin() for i in best_individuals[4]]
        end_best_individuals = [int(i + best_individuals[1] / self.step_width) for i in ini_best_individuals]

        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Position [m]')
        plt.plot(np.array(best_vel), best_pos, lw=1.0)
        plt.scatter(np.array(best_vel)[ini_best_individuals], np.array(best_pos)[ini_best_individuals],
                    s=10, facecolors='none', edgecolors='g', label='StartBurnTime')
        plt.scatter(np.array(best_vel)[end_best_individuals],
                    np.array(best_pos)[end_best_individuals],
                    s=10, facecolors='none', edgecolors='r', label='EndBurnTime')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_best(self, best_pos, best_vel, best_mass, best_thrust, best_individuals):
        fig_best, axs_best = plt.subplots(2, 2, constrained_layout=True)
        sq = 10
        time_best = np.arange(0, len(best_pos)) * self.step_width
        """
        ini_best_individuals: Initial ignition point
        end_best_individuals: End of burning process
        """
        ini_best_individuals = [np.array(np.abs(best_pos - i)).argmin() for i in best_individuals[4]]
        end_best_individuals = [int(i + best_individuals[1] / self.step_width) for i in ini_best_individuals]

        axs_best[0, 0].set_xlabel('Time [s]')
        axs_best[0, 0].set_ylabel('Position [m]')
        axs_best[0, 0].plot(time_best, best_pos, lw=1.0)
        axs_best[0, 0].scatter(time_best[ini_best_individuals], np.array(best_pos)[ini_best_individuals],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[0, 0].scatter(time_best[end_best_individuals],
                               np.array(best_pos)[end_best_individuals],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
        axs_best[0, 0].grid(True)
        axs_best[0, 0].legend()

        axs_best[0, 1].set_xlabel('Time [s]')
        axs_best[0, 1].set_ylabel('Velocity [m/s]')
        axs_best[0, 1].plot(time_best, np.array(best_vel), lw=1.0)
        axs_best[0, 1].scatter(time_best[ini_best_individuals], np.array(best_vel)[ini_best_individuals],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[0, 1].scatter(time_best[end_best_individuals],
                               np.array(best_vel)[end_best_individuals],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
        axs_best[0, 1].grid(True)
        # axs_best[0, 1].legend()

        axs_best[1, 0].set_xlabel('Time [s]')
        axs_best[1, 0].set_ylabel('Mass [kg]')
        axs_best[1, 0].plot(time_best, best_mass, lw=1.0)
        axs_best[1, 0].scatter(time_best[ini_best_individuals], np.array(best_mass)[ini_best_individuals],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[1, 0].scatter(time_best[end_best_individuals],
                               np.array(best_mass)[end_best_individuals],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
        axs_best[1, 0].grid(True)
        # axs_best[1, 0].legend()

        axs_best[1, 1].set_xlabel('Time [s]')
        axs_best[1, 1].set_ylabel('Thrust [N]')
        axs_best[1, 1].plot(time_best, 1e1 * np.array(best_thrust), lw=1.0)
        axs_best[1, 1].scatter(time_best[ini_best_individuals], 1e1 * np.array(best_thrust)[ini_best_individuals],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[1, 1].scatter(time_best[end_best_individuals],
                               1e1 * np.array(best_thrust)[end_best_individuals],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
        axs_best[1, 1].grid(True)
        # axs_best[1, 1].legend()

        plt.show()
        return

    def print_report(self, Hf, Vf, Mf, Tf, individual):
        print(':::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Max thrust by engine: ', individual[-1], ' [N]')
        print('Used fuel mass: ', Mf[0] - Mf[-1], ' [kg]')
        print('-------------------------------------------------')
        print('Switchinf functions')
        ini_best_individuals = [np.array(np.abs(Hf - i)).argmin() for i in individual[4]]
        end_best_individuals = [int(i + individual[1] / self.step_width) for i in individual]

        for i in range(len(individual[4])):
            print('sf_', i + 1, ': ', 'H_i = ', Hf[ini_best_individuals[i]], 'H_f = ', Hf[end_best_individuals[i]],
                  'V_i = ', -Vf[ini_best_individuals[i]], 'V_f = ', -Vf[end_best_individuals[i]])
        return


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    TUBULAR = 'tubular'
    BATES = 'bates'
    STAR = 'star'

    r0 = 5000
    v0 = 0
    m0 = 24
    Isp = 325
    g = -1.62
    ge = 9.807
    den_p = 1.74

    x1_0 = r0
    x2_0 = v0
    x1_f = 0
    x2_f = 0
    init_state = [[x1_0, x1_f],
                  [x2_0, x2_f],
                  m0]

    c_char = Isp * ge
    alpha_min = - g * m0 / c_char * 0.1
    alpha_max = alpha_min * 100.0
    n_min_thr, n_max_thr = 10, 10
    t_burn_min, t_burn_max = 2, 20
    range_variables = [['float', alpha_min, alpha_max], ['float', t_burn_min, t_burn_max],
                       ['int', n_max_thr], ['str', TUBULAR],
                       ['float_iter', x1_0, x1_f]]

    ga = GeneticAlgorithm(1, g, init_state, 30, 20, range_variables=range_variables)
