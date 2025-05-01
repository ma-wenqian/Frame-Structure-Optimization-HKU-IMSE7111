import frame_model as fm
import numpy as np
import pymc as pm
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import arviz as az
import argparse

def pymc_solution(rng, d, fitness_tag, max_U, size=None):
    TotalMass = fm.frame_model(d)
    max_u17, _, _ = fm.dynamic_analysis()
    if fitness_tag == 1:
        fitness = np.array(np.abs(max_u17 - max_U) + TotalMass)
    elif fitness_tag == 2:
        fitness = np.array((max_u17 - max_U)**2 + TotalMass)
    elif fitness_tag == 3:
        fitness = np.array([max_u17, TotalMass])    # return a numpy array
    else:
        raise ValueError("Invalid fitness type. Choose 'fitness1', 'fitness2', or 'fitness3'.")
    return fitness 


def bayesian_run(n_draws, n_chains, fitness_type="fitness1", max_U = 70.0, d_bounds=(150.0, 650.0)):
    print("Starting Bayesian Optimization...")
    # max_U = 70.0  # target maximum displacement
    if fitness_type == "fitness1":
        fitness_tag = 1
        observed_data = np.array([0.0])
    elif fitness_type == "fitness2":
        observed_data = np.array([0.0])
        fitness_tag = 2
    elif fitness_type == "fitness3":
        observed_data = np.array([max_U, 0.0])
        fitness_tag = 3 
    else:
        raise ValueError("Invalid fitness type. Choose 'fitness1', 'fitness2', or 'fitness3'.")

    print(f'fitness_type={fitness_type} n_draws={n_draws} n_chains={n_chains} observed_data={observed_data}')

    # Create results directory if it doesn't exist
    results_dir = os.path.join("results", "Bayesian_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    # bayesian optimization
    with pm.Model() as model:
        # Define priors for the parameters
        d = pm.Uniform('d', lower=d_bounds[0], upper=d_bounds[1], shape=8)  # Uniform distribution for d, dimension of the section
       
        # Define the simulator
        simulator = pm.Simulator("simulator", pymc_solution, d,fitness_tag, max_U, observed=observed_data)
        # Perform SMC sampling
        idata = pm.sample_smc(draws=n_draws, chains=n_chains)

        # Perform posterior predictive sampling
        idata.extend(pm.sample_posterior_predictive(idata))
    
    filename = f'{fitness_type}_n_chains={n_chains}_n_draws={n_draws}_'
    posterior_data=idata.posterior.d.values
    posterior_predictive_data=idata.posterior_predictive.simulator.values
    np.save(os.path.join(results_dir, filename + "posterior"), posterior_data)
    np.save(os.path.join(results_dir, filename + "posterior_predictive"), posterior_predictive_data)

    # Visualize the results
    print("Visualizing results...")
    
    # Posterior plot
    az.plot_posterior(idata)
    plt.savefig(os.path.join(results_dir, filename+"posterior_plot.png"))
    plt.close()

    # Summary statistics
    summary = az.summary(idata)
    summary.to_csv(os.path.join(results_dir, filename+"summary.csv"))

    solution = posterior_data.mean(axis=(0, 1))
    # Perform dynamic analysis with the best solution
    print("\nPerforming dynamic analysis with the best solution...")
    fm.result_save(solution, "ba-"+filename)
    
    print("\nBayesian optimization completed.")

if __name__ == "__main__":
    print("bayesian_solution.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_draws", type=int, default=100, help="Number of draws for the Bayesian model")
    parser.add_argument("--n_chains", type=int, default=4, help="Number of chains for the Bayesian model")
    parser.add_argument("--fitness_type", type=str, default="fitness1", help="Type of fitness function to use (fitness1, fitness2, fitness3)")
    parser.add_argument("--max_U", type=float, default=70.0, help="Target maximum displacement")
    args = parser.parse_args()

    n_draws = args.n_draws
    n_chains = args.n_chains
    fitness_type = args.fitness_type
    max_U = args.max_U
    d_bounds = (150.0, 650.0) # Lower and upper bounds for d

    
    bayesian_run(n_chains=n_chains, n_draws=n_draws, fitness_type=fitness_type, max_U=max_U, d_bounds=d_bounds)

    # Run the script with different parameters
    # python bayesian_solution.py --n_draws 100 --n_chains 4 --fitness_type fitness1
    # python bayesian_solution.py --n_draws 100 --n_chains 4 --fitness_type fitness2
    # python bayesian_solution.py --n_draws 100 --n_chains 4 --fitness_type fitness3








