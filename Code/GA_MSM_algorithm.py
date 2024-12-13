import time
import numpy as np
import copy
import random
from joblib import Parallel, delayed
from population_generator import populationGenerator
from crossover_operator import crossOverOperator
from fitness_operators import fitnessOperators
from mutation_operators import noiseMutator

class GAMSM:

"""
    The GAMSM class implements the original framework for GA-MSM by Faycal et al.
    This module is specifically designed for applications involving neural network weights as individuals in the population, 
    and it includes provisions for fitness diversity, elite preservation, and adaptive mutation strategies. 

    Key Features:
    - Population initialization and evolution over a specified number of generations.
    - Parallelized fitness computation for efficiency.
    - Elite preservation to retain top-performing individuals across generations.
    - Uniform crossover for generating new individuals from elite parents.
    - Noise-based mutation with decaying factor to balance exploration and exploitation.
    - Fitness novelty evaluation to maintain population diversity.

    This class is primarily intended for policy network weights optimization for models entering the FrozenLake environment.
"""

    def __init__(self, number_of_additional_mutations, population_number, elite_percentage, 
                 noise_type, number_of_generations, x):

        self.number_of_additional_mutations = number_of_additional_mutations
        self.population_number = population_number
        self.elite_percentage = elite_percentage
        self.noise_type = noise_type
        self.number_of_generations = number_of_generations
        self.generation_number = 0
        self.initial_population = populationGenerator.generate_population(self.population_number)
        self.average_population_fitness = []
        self.fitness_calculated = False
        self.x = x

    def run(self):
        
        population = self.initial_population
        
        start_time = time.time()

        while self.generation_number < self.number_of_generations:

            decay_factor = 1.0 / (1 + self.x*self.generation_number)
            
            if self.fitness_calculated == True:
                # no need to recalculate fitness if mutated_population used - already calculated its fitness before
                population_fitness = mutated_population_fitness
            else:
                # calculate population fitness in parallel, using available cores
                population_fitness = np.array(Parallel(n_jobs=-1)(delayed(fitnessOperators.compute_fitness)(individual) for individual in population))
            
            self.average_population_fitness.append(np.average(population_fitness)) # keeps track of average fitness of entire population

            # attach individuals to their fitness labels and sort by fitness in descending order (to extract elites later)
            labelled_individuals = list(zip(population, population_fitness))
            labelled_indv_descending_fitness = sorted(labelled_individuals, key=lambda x: x[1], reverse=True)

            # separate top 10% as elite performers
            elite_performers = labelled_indv_descending_fitness[:int(self.population_number * self.elite_percentage)]
            elite_individuals = [individual for individual, _ in elite_performers]

            # preserve the elites first in the new population
            mutated_population = [[] for _ in range(self.population_number)]
            mutated_population[:int(self.population_number * self.elite_percentage)]=copy.deepcopy(elite_individuals[:int(self.population_number*self.elite_percentage)])

            # index of the empty slots after elites
            range_num = int(self.population_number * self.elite_percentage)

            # crossover elites to create remaining population
            for j in range(range_num, self.population_number):
                parent1 = np.random.randint(0, range_num)
                parent2 = np.random.randint(0, range_num)
                mutated_population[j] = crossOverOperator.uniform_crossover(elite_individuals[parent1], elite_individuals[parent2])
            
            # mutate the population repeatedly number_of_additional_mutation number of times (except elites - they need to remain preserved)
            for _ in range(self.number_of_additional_mutations):
                for j in range(range_num, self.population_number):
                    mutated_population[j] = noiseMutator.mutate_with_noise(mutated_population[j], decay_factor, self.noise_type)
            
            # compute fitness for mutated population
            mutated_population_fitness = np.array(Parallel(n_jobs=-1)(delayed(fitnessOperators.compute_fitness)(individual) for individual in mutated_population))

            if sum(mutated_population_fitness) >= sum(population_fitness): # if mutated population performs better, then set as the population for next generation
                population = mutated_population
                self.fitness_calculated = True # no need to recalculate fitness when going through the loop again
            else:
                population_new = [[] for _ in range(self.population_number)]

                # create new population from elites
                for i in range(self.population_number):
                    parent1 = np.random.randint(0, len(elite_individuals))
                    parent2 = np.random.randint(0, len(elite_individuals))
                    population_new[i] = noiseMutator.mutate_with_noise(crossOverOperator.uniform_crossover(elite_individuals[parent1], elite_individuals[parent2]),decay_factor, self.noise_type)

                # calculate fitness for the new population using diversity measure
                population_new_fitness = np.array(Parallel(n_jobs=-1)(delayed(fitnessOperators.compute_fitness_novelty)(individual, mutated_population) for individual in population_new))

                # extract elite performers for new population based on diversity measure
                new_population_labelled = list(zip(population_new, population_new_fitness))
                labelled_new_indv_descending_fitness = sorted(new_population_labelled, key=lambda x: x[1], reverse=True)

                elite_performers = labelled_new_indv_descending_fitness[:int(self.population_number * self.elite_percentage)]
                elite_individuals = [individual for individual, _ in elite_performers]

                # create new population based on elite individuals (first preserve the new elites)
                population = [[] for _ in range(self.population_number)]
                population[:int(self.population_number * self.elite_percentage)] = copy.deepcopy(elite_individuals[:int(self.population_number * self.elite_percentage)])
                
                # create rest of population with new elites and mutate once
                for i in range(int(self.population_number * self.elite_percentage), self.population_number):
                    parent1 = np.random.randint(0, len(elite_individuals))
                    parent2 = np.random.randint(0, len(elite_individuals))
                    population[i] = noiseMutator.mutate_with_noise(crossOverOperator.uniform_crossover(elite_individuals[parent1], elite_individuals[parent2]), decay_factor, self.noise_type)

                self.fitness_calculated = False
                
            self.generation_number += 1

        end_time = time.time()

        total_time = end_time - start_time
        hours, minutes = divmod(total_time // 60, 60)

        print(f"Total Runtime: {int(hours)} hours and {int(minutes)} minutes.")
