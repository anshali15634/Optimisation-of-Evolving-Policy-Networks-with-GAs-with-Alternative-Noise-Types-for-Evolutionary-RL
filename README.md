
# COMP4082 - Autonomous Robotic Systems Coursework

## File Navigation

```
ARS-2425-Task2-20506330
└── Code
    ├── main-novelty.ipynb
    ├── main-RQ1.ipynb
    ├── main-RQ2.ipynb
    ├── graph_results.py
    ├── fitness_operators.py
    ├── crossover_operator.py
    ├── mutation_operators.py
    ├── GA_MSM_P_algorithm.py
    ├── GA_MSM_algorithm.py
    ├── GA_MSM_P_algorithm2.py
    ├── population_generator.py
    ├── noise_injection.py
    └── GA_MSM_algorithm2.py
└── Report
    └── Conference_Paper.pdf
└── Results & Processing
    ├── Gaussian_FrozenLake_GAMSM.xlsx
    ├── Gaussian_FrozenLake_GAMSMP.xlsx
    ├── Pink_FrozenLake_GAMSM.xlsx
    ├── Pink_FrozenLake_GAMSMP.xlsx
    ├── OU_FrozenLake_GAMSM.xlsx
    ├── OU_FrozenLake_GAMSMP.xlsx
    ├── scale0GAMSM.xlsx
    ├── scale0GAMSMP.xlsx
    ├── scale1GAMSM.xlsx
    ├── scale1GAMSMP.xlsx
    ├── scale2GAMSM.xlsx
    ├── scale2GAMSMP.xlsx
    ├── scale3GAMSM.xlsx
    ├── scale3GAMSMP.xlsx
    ├── novelty.xlsx
    └── Performance Metrics Calculations.ipynb


```

### System Configuration

The experiments were conducted on the following system specifications:

| Specification        | Details         |
|----------------------|-----------------|
| **System Architecture** | 64-bit         |
| **Processor**         | x86_64          |
| **Total RAM**         | 125.61 GB       |
| **Operating System**  | Linux           |
| **Python Version**    | 3.7.16          |


The .ipynb files are the Jupyter Notebooks which carry the experiments, additional documentation and final results. The .py files are custom modules created for the experiments.

***Quick Notes***
- Hyperparameters used are set as the original paper discussing the algorithm, this is outlined in the Conference Paper's Appendix.
- Tables of results are inside the Jupyter notebooks along with the experiments.
- Modules required for the experiment are specified in a code block in the beginning of each Jupyter notebook.
- Runtime taken for each experiment is detailed within the notebooks.

Below is a brief guide to the key files in the Code file.

## Notebooks (storing experiments and results)

### main-novelty.ipynb
This Jupyter notebook demonstrates the novelty added to the **GA-MSM (Genetic Algorithm with Multi Step Mutation)** algorithm, addressed as **GA-MSM-P (Genetic Algorithm with Multi Step Mutation with Preservation)**. It includes:
- **Algorithm Execution in 10 Runs**: GA-MSM-P and GA-MSM are executed for 10 runs on the Frozen Lake environment, their median scores for each generation across 10 runs are reported. 
- **Statistical Significance Tests**: Contains the code for running the Mann Whitney U test to validate the significance of the difference in performance in results produced by the GA-MSM and GA-MSM-P algorithmS.
- **Results and Visualizations**: Visualizations, statistical summary, and tabular version of results are included.

### main-RQ1.ipynb
This Jupyter notebook is dedicated to experiments for **Research Question 1 (RQ1)**, which investigates the effects of different noise types (Gaussian, Pink and OU) on GA-MSM-P and GA-MSM. The file includes:
- **Experimental Setup**: Code to set up and execute the experiments related to RQ1. 10 runs are performed for each noise type on the Frozen Lake environment, for each algorithm and median scores are reported.
- **Data Analysis**: Analyses experimental results using Mann Whitney U tests to check for significance in performance between noise types.
- **Results**: Outcomes of the experiments are visualised in tabular format and in graphs.

### main-RQ2.ipynb
This Jupyter notebook is dedicated to experiments for **Research Question 2 (RQ1)**, which investigates the effects of pink noise injection into the continuous input space for GA-MSM-P and GA-MSM using the CartPole environment. The file includes:
- **Experimental Setup**: Code to set up and execute the experiments related to RQ2. 5 runs are performed for each sigma value of pink noise injection, for each algorithm and median scores are reported.
- **Results**: Outcomes of the experiments are visualised in tabular format and in graphs.

### Performance Metrics Calculations.ipynb
This Jupyter Notebook calculates the number of generations taken to converge based on convergence criterion mentioned in Conference Paper.pdf, for both research questions. It uses the .xlsx files output from main-RQ1.ipynb and main-RQ2.ipynb notebooks.


## Custom Modules

### crossover_operator.py
Contains crossover operator function to perform simple uniform crossover for each GA.

### fitness_operators.py

Contains the following functions:

#### Model Creation for Different Environments
- `create_model(weight_set)`:
  - Constructs a neural network for the **FrozenLake environment**.
  - Architecture follows Core Paper A's GA-MSM description.

- `create_model2(weight_set)`:
  - Builds a neural network for the **CartPole environment**.
  - Tailored architecture for environment-specific tasks.

#### Fitness Evaluation
- `compute_fitness(weight_set)`:
  - Evaluates fitness in the **FrozenLake environment**.
  - Uses total reward as the fitness measure.

- `compute_fitness2(weight_set, pink_inject, inject_factor)`:
  - Computes fitness in the **CartPole environment**.
  - Includes optional **pink noise injection** to simulate state perturbations (for RQ2)

#### Diversity Fitness Calculation
- `compute_fitness_novelty(individual, mutated_population)`:
  - Measures novelty of the population by calculating the average Euclidean distance between an individual and a mutated population.
  - Promotes **directed diversity** in the population.

### population_generator.py

This module generates initial populations of randomised policy network weights for neural networks in reinforcement learning tasks. Two functions are provided to accommodate different environments with varying input and output space sizes, which alter the policy network's architecture.

#### Functions

- `glorot_uniform_range(n_in, n_out)`:
  - Calculates the Glorot (Xavier) uniform range based on the number of input and output neurons in a layer.
  - Ensures weights are initialized within a stable range.

- `generate_population(populationNumber)`:
  - Creates a population of neural network weight sets (300 weights per individual) for the **FrozenLake environment**.
  - Accounts for the architecture with input size 16 and output size 4, generating randomized weights across three layers.

- `generate_population2(populationNumber)`:
  - Generates a population of neural network weight sets (160 weights per individual) for the **CartPole environment**.
  - Adapts to the architecture with input size 4 and output size 2, producing randomized weights for three layers.

This module is essential for initializing diverse and compatible populations of neural network weights for ERL experiments.

### graph_results.py
Contains specific functions to display results.

### mutation_operators.py
Contains functions to add noise to the weight matrix for differently structured policy networks for each RQ.

### noise_injection.py

This module provides three noise injection mechanisms for use in Genetic Algorithms (GAs) to mutate neural network weights. The supported noise types are:

- **Pink Noise**: Time-correlated noise. Contains functions suitable for weight mutation and continuous input space noise injection.
- **Ornstein-Uhlenbeck (OU) Noise**: Temporal noise with mean-reversion characteristics, ideal for scenarios requiring continuity in noise samples.
- **Gaussian Noise**: Independently sampled noise, commonly used as the default mutation operator in GAs.

This module ensures that noise characteristics are adaptable, including support for altering decay factors dynamically to adjust the scale of noise over generations.

### GA-MSM_algorithm.py

Contains the algorithm GA-MSM, for the FrozenLake environment as dictated in the original framework by Faycal et al.

### GA_MSM_P_algorithm.py

Contains a novel modification to GA-MSM, for the FrozenLake environment, the statistical significance in its performance is judged in main-novelty.ipynb.

### GA_MSM_algorithm2.py

Adapted version of GA_MSM, but for the CartPole environment, and for pink noise injection into continuous input space.

### GA_MSM_P_algorithm2.py

Adapted version of GA_MSM_P but for the CartPole environment, and for pink noise injection into continuous input space.