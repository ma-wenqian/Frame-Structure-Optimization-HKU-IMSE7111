import frame_model as fm
import pygad
import numpy as np

# Define the fitness function for pygad
def ga_fitness_function(ga_instance, solution, solution_idx):
    d = solution[:8]  # First 8 elements are d
    # t = solution[8:]  # Last 8 elements are t
    t = np.array([0.05]*8)
    TotalMass = fm.frame_model(d,t)
    max_u17, _ = fm.dynamic_analysis()
    # Adjusted fitness function to target max_u17 close to 150 and minimize TotalMass
    fitness = 1.0 / (np.abs(max_u17 - 150) + TotalMass + 1e-6)  # Adding a small value to avoid division by zero
    # another optimization criteria
    # fitness = 1.0 / ((max_u17 - 150)**2 + TotalMass + 1e-6)
    return fitness


# Define the callback function to monitor the progress of the GA
def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

if __name__ == "__main__":
    print("Starting Genetic Algorithm Optimization...")
    # Define the parameter bounds
    d_bounds = (150, 600)  # Scaled by 300 as in bayesian_solution.py
    # t_bounds = (0.05, 0.8)

    # Define the number of genes (8 for d and 8 for t)
    num_genes = 8

    # Define the gene space for each parameter
    gene_space = [
        {'low': d_bounds[0], 'high': d_bounds[1]} for _ in range(8)
    ]

    # Initialize the GA instance
    ga_instance = pygad.GA(
        num_generations=150,
        num_parents_mating=4,
        fitness_func=ga_fitness_function,
        sol_per_pop=100,
        num_genes=num_genes,
        gene_space=gene_space,
        on_generation=on_gen,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        mutation_num_genes=1  # Explicitly set the number of genes to mutate
    )

    # Run the genetic algorithm
    ga_instance.run()

    # Retrieve the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    d_optimal = solution[:8]
    t_optimal = solution[8:]

    print("Optimal d:", d_optimal)
    print("Optimal fitness:", -solution_fitness)

    filename = 'genetic_result'
    ga_instance.save(filename=filename)
    ga_instance.plot_fitness()



