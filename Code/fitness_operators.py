import uuid
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import gymnasium as gym
from mutation_operators import noiseMutator
from noise_injection import PinkNoiseInjection

class fitnessOperators:

    """
    This class implements fitness evaluation methods and model generation for GA-MSM and GA-MSM-P.
    
    Specifically, it provides methods to:
    - Generate neural network models of fixed architectures for evaluating individuals in the environments FrozenLake and CartPole.
    - Compute the fitness of an individual based on either:
        - its performance in the specified environment (defined by Core Paper A, CW 1)
        - distance from underperforming mutated populations (defined by Core Paper A, CW 1)
    - Support extensions like noise mutation through pink noise injection to analyze its impact on GA convergence and generalization.

    These functionalities enable the evaluation and refinement of GA-MSM (Genetic Algorithm with Multi-State Memory) across multiple environments.
    """

    @staticmethod
    def create_model(weight_set):
        
        # define model architecture according Core Paper A's description of GA-MSM
        # this model is to be used for the FrozenLake environment
        # this function takes the set of policy network weights (individual from the population), sets it into the model architecture and returns the model.
        
        unique_id = str(uuid.uuid4())
        
        model = models.Sequential(name=f"model_{unique_id}")
        model.add(layers.Dense(10, activation='relu', kernel_initializer='glorot_uniform', input_shape=(16,), bias_initializer='zeros', name=f"dense1_{unique_id}"))
        model.add(layers.Dense(10, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name=f"dense2_{unique_id}"))
        model.add(layers.Dense(4, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros', name=f"dense3_{unique_id}"))
        model.compile(optimizer='adam', loss='mse')

        # set weights for each layer using provided weight set (individual from population)
        weights_layer1 = weight_set[:160].reshape((16, 10))
        weights_layer2 = weight_set[160:260].reshape((10, 10))
        weights_layer3 = weight_set[260:300].reshape((10, 4))
        
        bias_layer1 = np.zeros(10)
        bias_layer2 = np.zeros(10)
        bias_layer3 = np.zeros(4)

        model.layers[0].set_weights([weights_layer1, bias_layer1])
        model.layers[1].set_weights([weights_layer2, bias_layer2])
        model.layers[2].set_weights([weights_layer3, bias_layer3])
        
        return model

    @staticmethod
    def compute_fitness(weight_set):
        
        # this function computes the fitness of the model in the FrozenLake environment
        # the model is of structure 16-10-10-4 (as mentioned by Core Paper A)
        # computes fitness of individual based on performance in one episode of FrozenLake.
        
        model = fitnessOperators.create_model(weight_set)

        # initialize environment
        env = gym.make('FrozenLake-v1', is_slippery=False)
        total_reward = 0
        done = False
        truncated = False
        state = env.reset()[0]

        while not done and not truncated:
            state_one_hot = np.eye(16)[state].reshape(1, -1)  # convert state to one-hot encoding
            action_probs = model(state_one_hot, training=False)
            action = np.argmax(action_probs)  # get index of highest probability
            next_state, reward, done, truncated, _ = env.step(action)  # take step in environment
            total_reward += reward
            state = next_state

        env.close()
        del model
        tf.keras.backend.clear_session()

        return total_reward

    @staticmethod
    def compute_fitness_novelty(individual, mutated_population):
        
        # this fitness function computes how far the individual is from the mutated population
        # this is the diversity metric introduced by Core Paper A for GA-MSM.
        # uses Euclidean distance to calculate the average distance of the individual from the underperforming mutated population.
        
        total_distance = 0
        for mutated_individual in mutated_population:
            distance = np.linalg.norm(abs(mutated_individual - individual))
            total_distance += distance
        avg_distance = total_distance / len(mutated_population)
        return avg_distance

    @staticmethod
    def create_model2(weight_set):
        
        # this model architecture is for the CartPole environment
        # with structure 4-10-10-2 (only input and output layers changed to adapt to environment)
        # does the same as function compute_fitness() but for a different architecture
        
        unique_id = str(uuid.uuid4())
        
        model = models.Sequential(name=f"model_{unique_id}")
        model.add(layers.Dense(10, activation='relu', kernel_initializer='glorot_uniform', input_shape=(4,), bias_initializer='zeros', name=f"dense1_{unique_id}"))
        model.add(layers.Dense(10, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name=f"dense2_{unique_id}"))
        model.add(layers.Dense(2, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros', name=f"dense3_{unique_id}"))
        model.compile(optimizer='adam', loss='mse')

        # set weights for each layer using provided weight set (individual from population)
        weights_layer1 = weight_set[:40].reshape((4, 10))
        weights_layer2 = weight_set[40:140].reshape((10, 10))
        weights_layer3 = weight_set[140:160].reshape((10, 2))
        
        bias_layer1 = np.zeros(10)
        bias_layer2 = np.zeros(10)
        bias_layer3 = np.zeros(2)

        model.layers[0].set_weights([weights_layer1, bias_layer1])
        model.layers[1].set_weights([weights_layer2, bias_layer2])
        model.layers[2].set_weights([weights_layer3, bias_layer3])
        
        return model
        
    @staticmethod
    def compute_fitness2(weight_set, pink_inject, inject_factor):

        # computes fitness for models entering the Cartpole environment
        # for RQ2, pink noise into the environment is considered.
        # therefore an option pink_inject is included, to allow for noise injected into continuous input space.
        # inject_factor is the scale of the noise for the pink noise samples

        model = fitnessOperators.create_model2(weight_set)

        total_rewards = []  # List to store rewards of each episode

        for _ in range(10):  # Run for 10 episodes
            env = gym.make('CartPole-v0')
            
            # generate all time correlated pink noise samples beforehand for 200 timesteps
            if pink_inject:
                pink_noise_samples = PinkNoiseInjection.pink_noise_generate_samples((200,4), inject_factor)
                pink_sample_index = -1
            total_reward = 0
            done = False
            truncated = False
            state = env.reset()[0]
            
            while not done and not truncated:
                
                if pink_inject:
                    pink_sample_index +=1
                    state = state + pink_noise_samples[pink_sample_index]
                    
                    # clipping to ensure final state does not go above specified range allowed in documentation
                    state[0] = np.clip(state[0], -4.8, 4.8)
                    state[2] = np.clip(state[2], -0.418, 0.418)
                    
                action_probs = model(state.reshape(1, -1), training=False) # predict action with model
                action = np.argmax(action_probs)
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)  # Store total reward for this episode
            env.close()

        # Clean up the model
        del model
        tf.keras.backend.clear_session()

        # Return the average reward over 10 episodes
        average_reward = np.mean(total_rewards)
        return average_reward

