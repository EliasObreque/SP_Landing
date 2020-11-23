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
    def __init__(self, step_w, init_state, max_generation, n_variables, n_individuals, range_variables):
        self.init_state = init_state
        self.step_width = step_w
        self.Ah = 0.01
        self.Bh = 10.0
        self.Ch = 0.1
        self.selection_method = 'rank'
        self.current_cost = []
        self.max_generation = max_generation
        self.n_individuals = n_individuals
        self.n_variables = n_variables
        self.range_variables = range_variables
        self.ind_selected = []
        self.weight_crossover = 0.2
        self.prob_mut = 0.1
        self.thrust_comp = None
        self.population = []
        self.historical_cost = []
        self.create_first_population()
        self.dynamics_update = None
        self.c_char = 200

    def create_first_population(self):
        for k in range(self.n_individuals):
            individual = []
            for i in range(self.n_variables):
                if self.range_variables[i][0] == 'float':
                    temp = np.random.uniform(self.range_variables[i][1], self.range_variables[i][2])
                    individual.append(temp)
                elif self.range_variables[i][0] == 'int':
                    temp = np.random.randint(self.range_variables[i][1], self.range_variables[i][2])
                    individual.append(temp)
                elif self.range_variables[i][0] == 'str':
                    type_pro = np.random.randint(1, len(self.range_variables[i]))
                    temp = self.range_variables[i][type_pro]
                    individual.append(temp)
                else:
                    print()
            self.population.append(individual)
        return

    def optimize(self, dynamics_update, c_char):
        self.c_char = c_char
        self.dynamics_update = dynamics_update
        ge = 1
        self.ga_evaluate()
        print('Generation: ', ge, ', Cost: ', max(self.current_cost))
        best_index = self.current_cost.index(max(self.current_cost))
        percent = 0.8
        n_indiv_by_selec = round(self.n_individuals * percent)
        rest_ind = n_indiv_by_selec % 2
        n_indiv_by_selec -= rest_ind
        while ge < self.max_generation:  # or max(self.current_cost) <= 0.5:
            next_population = []
            crossover_arith = True
            for i in range(int(n_indiv_by_selec * 0.5)):
                index_parents = self.ga_selection(2)
                if crossover_arith:
                    descent1, descent2  = self.ga_crossover_arithmetic(index_parents[0], index_parents[1])
                    crossover_arith = False
                else:
                    descent1, descent2 = self.ga_crossover_coding(index_parents[0], index_parents[1])
                    crossover_arith = True
                new_generation1 = self.ga_mutation(descent1, True)
                new_generation2 = self.ga_mutation(descent2, True)
                next_population.append(new_generation1)
                next_population.append(new_generation2)
            if self.n_individuals - n_indiv_by_selec != 0:
                descent = self.ga_selection(self.n_individuals - n_indiv_by_selec, direct_method=True)
                next_population += descent
            self.population = next_population
            best_index = self.current_cost.index(max(self.current_cost))
            self.current_cost = []
            Hf, Vf, Mf, Tf = self.ga_evaluate()
            ge += 1
            if ge > self.max_generation:
                self.max_generation = ge
            print('Generation: ', ge, ', Cost: ', max(self.current_cost))
        self.plot_cost()
        best_pos = np.array(Hf[best_index])
        best_vel = np.array(Vf[best_index])
        best_mass = np.array(Mf[best_index])
        best_thrust = Tf[best_index]
        best_individuals = self.population[best_index]
        ini_best_individuals = [np.array(np.abs(best_pos - i)).argmin() for i in best_individuals[:-1]]
        end_best_individuals = [int(i + self.burn_time / self.step_width) for i in ini_best_individuals]

        self.hI.append([best_pos[ini_best_individuals], best_pos[end_best_individuals]])
        self.vI.append([best_vel[ini_best_individuals], best_vel[end_best_individuals]])
        self.tI.append(best_individuals[-1] * np.ones((1, self.n_thruster))[0])
        self.plot_best(best_pos, best_vel, best_mass, best_thrust, best_individuals)
        self.plot_state_vector(best_pos, best_vel, best_individuals)
        self.print_report(best_pos, best_vel, best_mass, best_thrust, best_individuals)
        return

    def plot_cost(self):
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Optimization function')
        plt.plot(np.arange(1, self.max_generation + 1), np.array(self.historical_cost)[:, 0], label='Min')
        plt.plot(np.arange(1, self.max_generation + 1), np.array(self.historical_cost)[:, 1], label='Max')
        plt.grid()
        plt.legend()
        plt.show()

    def calc_thrust(self, current_high, num_individual):
        population_k = self.population[num_individual]
        current_normalize_total_thrust = 0
        for i in range(self.n_thruster):
            ign_alt = population_k[i]
            if current_high <= ign_alt:
                self.thrust_comp[i].set_beta(1)
                self.thrust_comp[i].calc_thrust_mag(self.step_width * 1000)
            current_normalize_total_thrust += self.thrust_comp[i].current_mag_thrust_c
        return current_normalize_total_thrust * population_k[-1]

    def ga_evaluate(self):
        POS = []
        VEL = []
        MASS = []
        THR = []
        for i in range(self.n_individuals):
            k = 0
            x1 = [self.init_state[0][0]]
            x2 = [self.init_state[1][0]]
            x3 = [self.init_state[2][0]]
            thr = []
            comp_thrust = []
            end_condition = False
            for j in range(self.population[i][2]):
                comp_thrust.append(Thruster(self.step_width, max_burn_time=self.population[i][1],
                                            nominal_thrust=self.population[i][0] * self.c_char,
                                            type_propellant=self.population[i][3]))
                comp_thrust[j].set_lag_coef(0.15)

            while end_condition is False:

                next_state = self.dynamics_update(thr[k], x1[k], x2[k], x3[k])

                x1[i].append(next_state[0])
                x2[i].append(next_state[1])
                x3[i].append(next_state[2])

                k += 1
                if x1[i][k] < 0 and thr[i][k - 1] == 0.0:
                    end_condition = True
                    x1[i].pop(k)
                    x2[i].pop(k)
                    x3[i].pop(k)

            POS.append(x1)
            VEL.append(x2)
            MASS.append(x3)
            THR.append(thr)
        return POS, VEL, MASS, THR

    def ga_selection(self, n, direct_method=False):
        if direct_method is False:
            if self.selection_method == 'roulette':
                probability_selection = np.array(self.current_cost) / np.sum(self.current_cost)
                ind_seleccionado = np.random.choice(a=np.arange(self.n_individuals),
                                                    size=n,
                                                    p=list(probability_selection),
                                                    replace=True)
            elif self.selection_method == "rank":
                ranks = rankdata(-1 * np.array(self.current_cost))
                probabilidad_seleccion = 1 / ranks
                probabilidad_seleccion = probabilidad_seleccion / np.sum(probabilidad_seleccion)
                ind_seleccionado = np.random.choice(a=np.arange(self.n_individuals),
                                                    size=n,
                                                    p=list(probabilidad_seleccion),
                                                    replace=True)
            else:
                print('Select method')
                ind_seleccionado = []
        else:
            list_order = sorted(self.current_cost)
            value_select = list_order[-n:]
            ind_seleccionado = []
            for elem in value_select:
                index_select = np.where(self.current_cost == elem)[0][0]
                ind_seleccionado.append(self.population[int(index_select)])
        return ind_seleccionado

    def ga_crossover_coding(self, parents_1, parents_2):
        children_1 = []
        children_2 = []
        cut_position = np.random.randint(0, self.n_variables)
        children_1[:cut_position] = self.population[parents_1][:cut_position]
        children_1[cut_position:] = self.population[parents_2][cut_position:]
        children_2[:cut_position] = self.population[parents_2][:cut_position]
        children_2[cut_position:] = self.population[parents_1][cut_position:]
        return np.array(children_1), np.array(children_2)

    def ga_crossover_arithmetic(self, parents_1, parents_2):
        children_1 = []
        children_2 = []
        for i in range(self.n_thruster):
            chs1 = self.population[parents_1][i] * self.weight_crossover +\
                   (1 - self.weight_crossover) * self.population[parents_2][i]
            chs2 = self.population[parents_2][i] * self.weight_crossover + \
                   (1 - self.weight_crossover) * self.population[parents_1][i]
            children_1.append(chs1)
            children_2.append(chs2)
        children_1.append(self.population[parents_1][-1] * self.weight_crossover +
                          (1 - self.weight_crossover) * self.population[parents_2][-1])
        children_2.append(self.population[parents_2][-1] * self.weight_crossover +
                          (1 - self.weight_crossover) * self.population[parents_1][-1])
        return np.array(children_1), np.array(children_2)

    def ga_mutation(self, descent, mutate):
        if not mutate:
            return descent
        else:
            posiciones_mutadas = np.random.uniform(low=0, high=1, size=self.n_variables)
            posiciones_mutadas = posiciones_mutadas < self.prob_mut
            if np.all(~posiciones_mutadas):
                return descent
            else:
                descent = np.array(descent)
                descent[:self.n_thruster][posiciones_mutadas[:self.n_thruster]] += 0.5 * (
                        self.ini_pos - self.tar_pos) * np.random.normal(
                    0, 0.2,
                    self.n_thruster)[posiciones_mutadas[:self.n_thruster]]
                descent[:self.n_thruster][descent[:self.n_thruster] > self.ini_pos] = self.ini_pos
                descent[:self.n_thruster][descent[:self.n_thruster] < 0] = 0
                if posiciones_mutadas[-1]:
                    descent[-1] += self.thrust_range[1] * np.random.normal(0, 0.2)
                    if descent[-1] < self.thrust_range[0]:
                        descent[-1] = self.thrust_range[1]
                    elif descent[-1] > self.thrust_range[1]:
                        descent[-1] = self.thrust_range[1]
                    else:
                        return descent
        return descent

    def calc_cost_function(self, pos_, vel_, mass_, thrust, k):
        error_pos = pos_[-1] - self.tar_pos
        error_vel = vel_[-1] - self.tar_vel
        max_engine = 2
        current_thrust = self.population[k][-1]
        current_engine = np.max(thrust) / current_thrust
        if np.min(np.array(pos_)) < 0.0:
            return (max_engine / current_engine) * 0.01 / (1 + self.Ah * np.abs(error_pos) + self.Bh * np.abs(error_vel))
        else:
            return 1 / (1 + self.Ah * np.abs(error_pos) + self.Bh * np.abs(error_vel))

    def calc_force_torque(self):
        self.force_ = 0

    def get_force_i(self):
        return self.force_

    def plot_initial_population(self):
        plt.figure()
        plt.subplot(221)
        plt.plot(self.total_thrust[0] * .001, lw=1.0)
        plt.grid()
        plt.subplot(222)
        plt.plot(self.total_thrust[1] * .001, lw=1.0)
        plt.grid()
        plt.subplot(223)
        plt.plot(self.total_thrust[2] * .001, lw=1.0)
        plt.grid()
        plt.subplot(224)
        plt.plot(self.total_thrust[3] * .001, lw=1.0)
        plt.grid()
        plt.show()

    def plot_state_vector(self, best_pos, best_vel, best_individuals):
        plt.figure()
        ini_best_individuals = [np.array(abs(best_pos - i)).argmin() for i in best_individuals[:-1]]
        end_best_individuals = [int(i + self.burn_time / self.step_width) for i in ini_best_individuals]
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
        ini_best_individuals = [np.array(np.abs(best_pos - i)).argmin() for i in best_individuals[:-1]]
        end_best_individuals = [int(i + self.burn_time/self.step_width) for i in ini_best_individuals]
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
        axs_best[0, 1].legend()

        axs_best[1, 0].set_xlabel('Time [s]')
        axs_best[1, 0].set_ylabel('Mass [kg]')
        axs_best[1, 0].plot(time_best, best_mass, lw=1.0)
        axs_best[1, 0].scatter(time_best[ini_best_individuals], np.array(best_mass)[ini_best_individuals],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[1, 0].scatter(time_best[end_best_individuals],
                               np.array(best_mass)[end_best_individuals],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
        axs_best[1, 0].grid(True)
        axs_best[1, 0].legend()

        axs_best[1, 1].set_xlabel('Time [s]')
        axs_best[1, 1].set_ylabel('Thrust [kN]')
        axs_best[1, 1].plot(time_best, 0.001 * np.array(best_thrust), lw=1.0)
        axs_best[1, 1].scatter(time_best[ini_best_individuals], 0.001 * np.array(best_thrust)[ini_best_individuals],
                               s=sq, facecolors='none', edgecolors='g', label='StartBurnTime')
        axs_best[1, 1].scatter(time_best[end_best_individuals],
                               0.001 * np.array(best_thrust)[end_best_individuals],
                               s=sq, facecolors='none', edgecolors='r', label='EndBurnTime')
        axs_best[1, 1].grid(True)
        axs_best[1, 1].legend()

        plt.show()
        return

    def print_report(self, Hf, Vf, Mf, Tf, individual):
        print(':::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Max thrust by engine: ', individual[-1], ' [N]')
        print('Used fuel mass: ', Mf[0] - Mf[-1], ' [kg]')
        print('-------------------------------------------------')
        print('Switchinf functions')
        ini_best_individuals = [np.array(abs(Hf - i)).argmin() for i in individual[:-1]]
        end_best_individuals = [int(i + self.burn_time / self.step_width) for i in ini_best_individuals]
        for i in range(self.n_thruster):
            print('sf_', i+1, ': ', 'H_i = ', Hf[ini_best_individuals[i]], 'H_f = ', Hf[end_best_individuals[i]],
                  'V_i = ', -Vf[ini_best_individuals[i]], 'V_f = ', -Vf[end_best_individuals[i]])
        return
