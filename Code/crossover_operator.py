import numpy as np
import random

class crossOverOperator:
    """
    The original framework for GA-MSM requires the use either Directed Crossover or Uniform Crossover.
    This study uses Uniform Crossover, for both GA-MSM and GA-MSM-P.
    """
    @staticmethod
    def uniform_crossover(parent1, parent2):
      # randomly choose weights from parents with 50-50 probability and append to child
        assert parent1.shape == parent2.shape
        child = []
        for i in range(len(parent1)):
            number = random.randint(1,2)
            if number == 1:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return np.array(child)