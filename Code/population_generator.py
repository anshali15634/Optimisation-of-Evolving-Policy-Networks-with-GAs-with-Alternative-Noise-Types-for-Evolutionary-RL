import numpy as np

class populationGenerator:

    """
    A static class for generating initial populations of neural network weight sets for use in GAs.
    This implementation is tailored to reinforcement learning tasks requiring specific network architectures, 
    FrozenLake and CartPole.

    Features:
    - Glorot Uniform Range Calculation: Ensures that weights are initialized within the optimal range for stability using the Glorot (Xavier) distribution.
    - Environment-Specific Populations: Provides two methods, `generate_population` and `generate_population2`, to create populations for distinct neural network architectures:
        - `generate_population`: Creates weight sets for a 16-10-10-4 network (for FrozenLake).
        - `generate_population2`: Creates weight sets for a 4-10-10-2 network (for CartPole).
    - Population Structure: Each individual in the population is represented as a flattened array of weights, ensuring compatibility with GAs.

    Implementation Details:
    - Weight ranges for each layer are dynamically computed using the Glorot uniform formula.
    - Individuals are initialized by sampling uniformly within these calculated ranges, ensuring balanced initialization.
    - Populations are returned as NumPy arrays for efficient manipulation during evolutionary processes.
    """

    @staticmethod
    def glorot_uniform_range(n_in, n_out):
        # each weight should be chosen randomly from a calculated range, using the Xavier (Glorot) distribution
        limit = np.sqrt(6 / (n_in + n_out))
        return limit

    @staticmethod
    def generate_population(populationNumber):
        # each vector contains 300 random weights
        # generate_population(populationNumber) will create an entire population of neural network weight sets to plug into the policy network
        vectors = []  # will store population of sets of neural network weights
        for i in range(populationNumber):
            vector = [] # will store one set of policy neural network weights (300 weights in total)
            # calculating the Glorot uniform range for each layer
            range160 = populationGenerator.glorot_uniform_range(16, 10)
            range100 = populationGenerator.glorot_uniform_range(10, 10)
            range40 = populationGenerator.glorot_uniform_range(10, 4)
            # randomize weights based on calculated range from the number of input neurons and output neurons
            vector.extend(np.random.uniform(-range160, range160, 160))
            vector.extend(np.random.uniform(-range100, range100, 100))
            vector.extend(np.random.uniform(-range40, range40, 40))
            # append the vector (one individual) to the population
            vectors.append(vector)

        vectors = np.array(vectors) # converted to numpy array - easier to calculate
        return vectors
    
    @staticmethod
    def generate_population2(populationNumber):
        # each vector is contains 160 random weights
        vectors = []  # will store population of sets of neural network weights
        for i in range(populationNumber):
            vector = [] # will store one set of policy neural network weights (160 weights in total)
            # calculating the Glorot uniform range for each layer
            range40 = populationGenerator.glorot_uniform_range(4, 10)
            range100 = populationGenerator.glorot_uniform_range(10, 10)
            range20 = populationGenerator.glorot_uniform_range(10, 2)
            # randomize weights based on calculated range from the number of input neurons and output neurons
            vector.extend(np.random.uniform(-range40, range40, 40))
            vector.extend(np.random.uniform(-range100, range100, 100))
            vector.extend(np.random.uniform(-range20, range20, 20))
            # append the vector (one individual) to the population
            vectors.append(vector)

        vectors = np.array(vectors) # converted to numpy array - easier to calculate
        return vectors
    
        