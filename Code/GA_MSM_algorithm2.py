import time
import numpy as np
import copy
import random
from joblib import Parallel, delayed
from population_generator import populationGenerator
from crossover_operator import crossOverOperator
from fitness_operators import fitnessOperators
from mutation_operators import noiseMutator

class GAMSM2:

    # This implementation of GA-MSM is specifically adapted to accommodate the CartPole environment and facilitate noise injection in continuous action spaces. 
    # Given the distinct policy network architecture required for CartPole, the fitness evaluation methods and mutation operations have been structured accordingly to ensure compatibility.
    # two additional parameters are added - pink inject, a boolean which flags whether pink noise should be injected into the input space, and inject factor, which specifies the scale of the pink noise.


    def __init__(self, number_of_additional_mutations, population_number, elite_percentage, 
                 noise_type, number_of_generations, x, pink_inject, inject_factor):

        self.number_of_additional_mutations = number_of_additional_mutations
        self.population_number = population_number
        self.elite_percentage = elite_percentage
        self.noise_type = noise_type
        self.number_of_generations = number_of_generations
        self.generation_number = 0
        self.initial_population = populationGenerator.generate_population2(self.population_number)
        self.average_population_fitness = []
        self.fitness_calculated = False
        self.x = x
        self.pink_inject = pink_inject
        self.inject_factor = inject_factor

    def run(self):
        
        population = self.initial_population
        
        start_time = time.time()

        while self.generation_number < self.number_of_generations:

            decay_factor = 1.0 / (1 + self.x*self.generation_number)
            
            if self.fitness_calculated == True:
                population_fitness = mutated_population_fitness # no need to recalculate
            else:
                # calculate population fitness in parallel
                population_fitness = np.array(Parallel(n_jobs=-1)(delayed(fitnessOperators.compute_fitness2)(individual, self.pink_inject, self.inject_factor) for individual in population))
            self.average_population_fitness.append(np.average(population_fitness))

            # label individuals with their fitness and sort by fitness
            labelled_individuals = list(zip(population, population_fitness))
            labelled_indv_descending_fitness = sorted(labelled_individuals, key=lambda x: x[1], reverse=True)

            # separate top 10% as elite performers
            elite_performers = labelled_indv_descending_fitness[:int(self.population_number * self.elite_percentage)]
            elite_individuals = [individual for individual, _ in elite_performers]

            # create mutated population
            mutated_population = [[] for _ in range(self.population_number)]
            mutated_population[:int(self.population_number * self.elite_percentage)]=copy.deepcopy(elite_individuals[:int(self.population_number*self.elite_percentage)])

            # index of the empty slots after elites
            range_num = int(self.population_number * self.elite_percentage)

            # crossover remaining population (not elite)
            for j in range(range_num, self.population_number):
                parent1 = np.random.randint(0, range_num)
                parent2 = np.random.randint(0, range_num)
                mutated_population[j] = crossOverOperator.uniform_crossover(elite_individuals[parent1], elite_individuals[parent2])
            
            # mutate the population repeatedly number_of_additional_mutation number of times
            for _ in range(self.number_of_additional_mutations):
                for j in range(range_num, self.population_number):
                    mutated_population[j] = noiseMutator.mutate_with_noise2(mutated_population[j], decay_factor, self.noise_type)
            
            # fitness for mutated population
            mutated_population_fitness = np.array(Parallel(n_jobs=-1)(delayed(fitnessOperators.compute_fitness2)(individual, self.pink_inject, self.inject_factor) for individual in mutated_population))

            if sum(mutated_population_fitness) >= sum(population_fitness):
                population = mutated_population
                self.fitness_calculated = True
            else:
                population_new = [[] for _ in range(self.population_number)]

                # create new population from elites
                for i in range(self.population_number):
                    parent1 = np.random.randint(0, len(elite_individuals))
                    parent2 = np.random.randint(0, len(elite_individuals))
                    population_new[i] = noiseMutator.mutate_with_noise2(crossOverOperator.uniform_crossover(elite_individuals[parent1], elite_individuals[parent2]),decay_factor, self.noise_type)

                # calculate fitness for the new population using diversity measure
                population_new_fitness = np.array(Parallel(n_jobs=-1)(delayed(fitnessOperators.compute_fitness_novelty)(individual, mutated_population) for individual in population_new))

                # elite performers for new population
                new_population_labelled = list(zip(population_new, population_new_fitness))
                labelled_new_indv_descending_fitness = sorted(new_population_labelled, key=lambda x: x[1], reverse=True)

                elite_performers = labelled_new_indv_descending_fitness[:int(self.population_number * self.elite_percentage)]
                elite_individuals = [individual for individual, _ in elite_performers]

                # create new population based on elite individuals
                population = [[] for _ in range(self.population_number)]
                population[:int(self.population_number * self.elite_percentage)] = copy.deepcopy(elite_individuals[:int(self.population_number * self.elite_percentage)])

                for i in range(int(self.population_number * self.elite_percentage), self.population_number):
                    parent1 = np.random.randint(0, len(elite_individuals))
                    parent2 = np.random.randint(0, len(elite_individuals))
                    population[i] = noiseMutator.mutate_with_noise2(crossOverOperator.uniform_crossover(elite_individuals[parent1], elite_individuals[parent2]), decay_factor, self.noise_type)

                self.fitness_calculated = False
                
            self.generation_number += 1

        end_time = time.time()

        total_time = end_time - start_time
        hours, minutes = divmod(total_time // 60, 60)

        print(f"Total Runtime: {int(hours)} hours and {int(minutes)} minutes.")
