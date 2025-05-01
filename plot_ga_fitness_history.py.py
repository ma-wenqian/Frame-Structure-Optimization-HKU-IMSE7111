import numpy as np
import matplotlib.pyplot as plt
import os

# Read arrays from txt files using numpy
fitness1_num_gen_30_sol_per_pop_10 = np.loadtxt(os.path.join('results', 'ga_results', 'fitness1_num_gen=30_sol_per_pop=10.txt'), skiprows=5)
fitness1_num_gen_150_sol_per_pop_100 = np.loadtxt(os.path.join('results', 'ga_results', 'fitness1_num_gen=150_sol_per_pop=100.txt'), skiprows=5)
fitness2_num_gen_30_sol_per_pop_10 = np.loadtxt(os.path.join('results', 'ga_results', 'fitness2_num_gen=30_sol_per_pop=10.txt'), skiprows=5)
fitness2_num_gen_150_sol_per_pop_100 = np.loadtxt(os.path.join('results', 'ga_results', 'fitness2_num_gen=150_sol_per_pop=100.txt'), skiprows=5)

# First figure: first two arrays
plt.figure(figsize=(8,5))
plt.plot(fitness1_num_gen_30_sol_per_pop_10, label='num gen=30, sol per pop=10')
plt.plot(fitness1_num_gen_150_sol_per_pop_100, label='num gen=150, sol per pop=100')
plt.title('GA Fitness 1 History')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('results', 'ga_results', 'fitness1_history.png'))

# Second figure: second two arrays
plt.figure(figsize=(8,5))
plt.plot(fitness2_num_gen_30_sol_per_pop_10, label='num gen=30, sol per pop=10')
plt.plot(fitness2_num_gen_150_sol_per_pop_100, label='num gen=150, sol per pop=100')
plt.title('GA Fitness 2 History')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('results', 'ga_results', 'fitness2_history.png'))