import matplotlib.pyplot as plt
import numpy as np

class graphResults:
"""
    A static class for visualizing the performance of GAs across multiple generations. 
    This class provides specific utility methods to plot results for one or more algorithms or noise configurations.
"""
    @staticmethod
    def display_results(gen_num, x_label, y_label, plot_title, results):
        
        plt.figure(figsize=(30,15))
        plt.xticks(np.arange(0,gen_num+1, 5))
        plt.plot(range(0, gen_num), results, marker='o', linestyle='-', color='b')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)

        plt.grid(True)
        plt.show()
    
    def display_double_results(gen_num, x_label, y_label, plot_title, results1, results2):
        
        plt.figure(figsize=(15,10))
        plt.xticks(np.arange(0,gen_num+1, 5))
        plt.plot(range(0, gen_num), results1, marker='o', linestyle='-', color='b', label = 'GA-MSM')
        plt.plot(range(0, gen_num), results2, marker='o', linestyle='-', color='r', label = 'GA-MSM-P')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)

        plt.grid(True)
        plt.legend()
        plt.show()
    
    def display_triple_results(gen_num, x_label, y_label, plot_title, results1, results2, results3):
        
        plt.figure(figsize=(15,10))
        plt.xticks(np.arange(0,gen_num+1, 5))
        plt.plot(range(0, gen_num), results1, marker='o', linestyle='-', color='g', label = 'Gaussian')
        plt.plot(range(0, gen_num), results2, marker='o', linestyle='-', color='#FF69B4', label = 'Pink')
        plt.plot(range(0, gen_num), results3, marker='o', linestyle='-', color='b', label = 'OU')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)

        plt.grid(True)
        plt.legend()
        plt.show()

    def display_four_results(gen_num, x_label, y_label, plot_title, results1, results2, results3, results4, mark):
        
        plt.figure(figsize=(15,10))
        plt.xticks(np.arange(0,gen_num+1, 5))
        plt.plot(range(0, gen_num), results1, marker=mark, linestyle='-', color='#A53E76', label = 'σ = 1.0')
        plt.plot(range(0, gen_num), results2, marker=mark, linestyle='-', color='#E2619F', label = 'σ = 0.5')
        plt.plot(range(0, gen_num), results3, marker=mark, linestyle='-', color='#E5A036', label = 'σ = 0.1')
        plt.plot(range(0, gen_num), results4, marker=mark, linestyle='-', color='#13000A', label = 'No Pink Noise Injection')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)

        plt.grid(True)
        plt.legend()
        plt.show()
        