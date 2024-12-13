import numpy as np
from population_generator import populationGenerator
import colorednoise as cn
from noise_injection import PinkNoiseInjection, OUNoiseInjectionParameterSpace, GaussianNoiseParameterSpace
class noiseMutator:
    """
    A static class for applying noise-based mutations to neural network weights, tailored for different reinforcement 
    learning environments, specifically FrozenLake and CartPole. The class accommodates the distinct policy network 
    architectures required for each environment through two mutation methods: `mutate_with_noise` for FrozenLake and 
    `mutate_with_noise2` for CartPole. 

    Features:
    - Supports three noise types: Gaussian, Pink, and Ornstein-Uhlenbeck (OU) noise.
    - Dynamically generates and injects noise into neural network weights for mutation.
    - Ensures mutated weights remain within the Glorot uniform distribution range using clipping for stability.
    - Facilitates time-correlated noise injection, especially for pink and OU noise, to maintain realistic perturbations 
      across weights.

    Implementation Details:
    - Static ranges for weight clipping (e.g., `RANGE160`, `RANGE100`) are computed using the Glorot uniform distribution 
      based on layer dimensions.
    - Mutated weights are reshaped and clipped layer-wise to conform to the original network architecture.
    - Time-correlated noise, such as OU and pink, is generated as sequential samples for consecutive weights to preserve temporal 
      coherence.
    - Output weights are recombined into a flattened array for compatibility with the Genetic Algorithm pipeline.
    """

    RANGE160 = populationGenerator.glorot_uniform_range(16, 10)
    RANGE100 = populationGenerator.glorot_uniform_range(10, 10)
    RANGE40 = populationGenerator.glorot_uniform_range(10, 4)
    RANGE20 = populationGenerator.glorot_uniform_range(10, 2)

    @staticmethod
    def mutate_with_noise(individual, decay_factor, noise_type):
        
        # select noise generation method
        if noise_type == 'gaussian':
            noise_array = GaussianNoiseParameterSpace.gaussian_noise(individual.shape, decay_factor)
        elif noise_type == 'pink':
            noise_array = PinkNoiseInjection.pink_noise(individual, decay_factor)
        elif noise_type == 'ou':
            ou_noise = OUNoiseInjectionParameterSpace(1) # 1D samples array
            noise_array = np.zeros(individual.shape)
            ou_noise.alter_decay_factor(decay_factor)
            for i in range(len(individual)):
                noise_array[i] = ou_noise.sample() # time correlated samples for consecutive weights
            del ou_noise
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        individual = individual + noise_array
        
        # rearrange noise added weights for each layer
        weights_layer1 = individual[:160].reshape(16,10)
        weights_layer2 = individual[160:260].reshape(10,10)
        weights_layer3 = individual[260:300].reshape(10,4)

        # ensure weights are within stable range
        final_weights_layer1 = np.clip(weights_layer1, -noiseMutator.RANGE160, noiseMutator.RANGE160)
        final_weights_layer2 = np.clip(weights_layer2, -noiseMutator.RANGE100, noiseMutator.RANGE100)
        final_weights_layer3 = np.clip(weights_layer3, -noiseMutator.RANGE40, noiseMutator.RANGE40)

        # Concatenate mutated weights back into a single 1D array
        mutated_individual = np.concatenate([
            final_weights_layer1.flatten(),
            final_weights_layer2.flatten(),
            final_weights_layer3.flatten()
        ])
        return mutated_individual

    @staticmethod
    def mutate_with_noise2(individual, decay_factor, noise_type):
        
        # select noise generation method
        if noise_type == 'gaussian':
            noise_array = GaussianNoiseParameterSpace.gaussian_noise(individual.shape, decay_factor)
        elif noise_type == 'pink':
            noise_array = PinkNoiseInjection.pink_noise(individual, decay_factor)
        elif noise_type == 'ou':
            noise_array = np.zeros((individual.shape, size))
            ou_noise.alter_decay_factor(decay_factor)
            for i in range(len(individual)):
                noise_array[i] = ou_noise.sample() # time correlated samples for consecutive weights
            ou_noise.reset()
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        individual = individual + noise_array
        
        # rearrange weights for each layer
        weights_layer1 = individual[:40].reshape(4,10)
        weights_layer2 = individual[40:140].reshape(10,10)
        weights_layer3 = individual[140:160].reshape(10,2)
        
        # Add noise and ensure weights are within required range
        final_weights_layer1 = np.clip(weights_layer1, -noiseMutator.RANGE40, noiseMutator.RANGE40)
        final_weights_layer2 = np.clip(weights_layer2, -noiseMutator.RANGE100, noiseMutator.RANGE100)
        final_weights_layer3 = np.clip(weights_layer3, -noiseMutator.RANGE20, noiseMutator.RANGE20)

        # Concatenate mutated weights into a single 1D array
        mutated_individual = np.concatenate([
            final_weights_layer1.flatten(),
            final_weights_layer2.flatten(),
            final_weights_layer3.flatten()
        ])
        return mutated_individual