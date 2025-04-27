import frame_model as fm
import numpy as np
import pymc as pm
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import arviz as az

def pymc_solution(rng, d, size=None):
    t = np.array([0.05]*8)
    TotalMass = fm.frame_model(d, t)
    max_u17, _ = fm.dynamic_analysis()
    response = np.array([max_u17, TotalMass])
    return response # return a numpy array


if __name__ == "__main__":
    print("Starting Bayesian Optimization...")
    n_draws = 100
    n_chains = 4
    # Create results directory if it doesn't exist
    results_dir = "Bayesian_optimization_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    observed_data = np.array([150.0, 0.0])  
    # bayesian optimization
    with pm.Model() as model:
        # Define priors for the parameters
        d = pm.Uniform('d', lower=0.6*300, upper=4*300, shape=8)  # Uniform distribution for d, dimension of the section
        # t = pm.Uniform('t', lower=0.1, upper=0.8, shape=8)  # Uniform distribution for t, percentage of d

        # Define the simulator
        simulator = pm.Simulator("simulator", pymc_solution, params=[d], observed=observed_data)
        # Perform SMC sampling
        idata = pm.sample_smc(draws=n_draws, chains=n_chains)

        # Perform posterior predictive sampling
        idata.extend(pm.sample_posterior_predictive(idata))

        # Visualize the results
        print("Visualizing results...")

        # Trace plot
        az.plot_trace(idata)
        plt.savefig(os.path.join(results_dir, "trace_plot.png"))
        plt.close()

        # Posterior plot
        az.plot_posterior(idata)
        plt.savefig(os.path.join(results_dir, "posterior_plot.png"))
        plt.close()

        # Summary statistics
        summary = az.summary(idata)
        summary.to_csv(os.path.join(results_dir, "summary.csv"))

        # Posterior predictive check
        az.plot_ppc(idata)
        plt.savefig(os.path.join(results_dir, "ppc_plot.png"))
        plt.close()

        # Perform diagnostics
        print("Running diagnostics...")
        rhat = az.rhat(idata)
        ess = az.ess(idata)
        print("Rhat values:")
        print(rhat)
        print("Effective Sample Size (ESS):")
        print(ess)







