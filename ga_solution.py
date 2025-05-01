import frame_model as fm
import pygad
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Define the fitness function for pygad
def ga_fitness_function(ga_instance, solution, solution_idx):
    d = solution[:8]  # First 8 elements are d
    TotalMass = fm.frame_model(d)
    max_u17, _, _ = fm.dynamic_analysis()
    # Adjusted fitness function to target max_u17 close to 150 and minimize TotalMass
    if fitness_type == "fitness1":
        fitness = 1.0 / (np.abs(max_u17 - 70) + TotalMass + 1e-6)  # Adding a small value to avoid division by zero
    elif fitness_type == "fitness2":
        fitness = 1.0 / ((max_u17 - 70)**2 + TotalMass + 1e-6)
    else:
        raise ValueError("Invalid fitness type. Choose 'fitness1' or 'fitness2'.")
    return fitness

# Define the callback function to monitor the progress of the GA
def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

# Define the function to run the GA
def ga_run(d_bounds, num_generations, sol_per_pop, fitness_type="fitness1"):
    print(f'fitness_type={fitness_type}')
    filename= f'{fitness_type}_num_gen={num_generations}_sol_per_pop={sol_per_pop}'

    # Define the number of genes (8 for d and 8 for t)
    num_genes = 8

    # Define the gene space for each parameter
    gene_space = [
        {'low': d_bounds[0], 'high': d_bounds[1]} for _ in range(8)
    ]

    # Initialize the GA instance
    ga_instance = pygad.GA( 
        num_generations=num_generations,
        num_parents_mating=4,
        fitness_func=ga_fitness_function,
        sol_per_pop=sol_per_pop,
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


    resultfile = os.path.join("results", "ga_results", filename)

    # extract the best solution and its fitness
    solution, solution_fitness, _ = ga_instance.best_solution()
    fitness_history = np.array(ga_instance.best_solutions_fitness)

    np.savetxt(resultfile+'.txt', fitness_history, header="fitness_history", comments=f'#{filename} \n# solution={np.array(solution)}\n# solution_fitness={solution_fitness}\n', delimiter=',')
    ga_instance.save(filename=resultfile)  # Save the GA instance to a file
    # Plot the fitness history
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history, marker='o')
    plt.title('GA Fitness History')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.savefig(resultfile+'_fitness_history.png')
    plt.close()

    print("\nOptimal d:", solution)
    print("Optimal fitness:", solution_fitness)

    # Perform dynamic analysis with the best solution
    print("\nPerforming dynamic analysis with the best solution...")
    fm.result_save(solution, filename)
    print("\n", filename, "Genetic Algorithm optimization completed.")




if __name__ == "__main__":
    print("Starting Genetic Algorithm Optimization...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_generations", type=int, default=30, help="Number of generations for the GA")
    parser.add_argument("--sol_per_pop", type=int, default=10, help="Number of solutions in the population")
    parser.add_argument("--fitness_type", type=str, default="fitness1", help="Type of fitness function to use")
    args = parser.parse_args()

    d_bounds=(150.0, 650.0)
    num_generations=args.num_generations
    sol_per_pop=args.sol_per_pop
    fitness_type=args.fitness_type

    ga_run(d_bounds=d_bounds, num_generations=num_generations, sol_per_pop=sol_per_pop, fitness_type=fitness_type)


    # ga_run(num_generations=30, sol_per_pop=10, fitness_type="fitness1")
    # 
    # Run the script with different parameters
    # python ga_solution.py --num_generations 30 --sol_per_pop 10 --fitness_type fitness1
    # python ga_solution.py --num_generations 30 --sol_per_pop 10 --fitness_type fitness2
    # python ga_solution.py --num_generations 150 --sol_per_pop 100 --fitness_type fitness2



