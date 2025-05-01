# Optimizing Frame Structure Resilience to Seismic Load Using Bayesian Optimization and Genetic Algorithms

This repository contains the code for the group project of the HKU course IMSE7111. The project focuses on optimizing the resilience of frame structures to seismic loads using Bayesian Optimization and Genetic Algorithms.

## Project Structure

- `frame_model.py`: Frame model implementation using OpenSeesPy, including dynamic analysis and result saving.
- `bayesian_solution.py`: Bayesian Optimization for structural optimization (using PyMC). Results and plots are saved in `results/Bayesian_results/`.
- `ga_solution.py`: Genetic Algorithm optimization (using PyGAD). Results and plots are saved in `results/ga_results/`.
- `Summarize_result.py`: Summarizes and compares optimization results.
- `plot_ga_fitness_history.py`: Plots GA fitness history from result files.
- `elcentro.txt`: Seismic load data used in the analysis.
- `results/`: Contains all output files, including optimization logs, summaries, and plots.

## Environment Setup

To run the code in this repository, ensure you have the following environment:

- **Python Version**: 3.8 or later
- **Required Libraries**:
  - `openseespy`
  - `numpy`
  - `opsvis`
  - `matplotlib`
  - `pymc`
  - `pygad`

You can install the required libraries using the following command:

```bash
pip install openseespy numpy opsvis matplotlib pymc pygad
```

If you are using a conda environment, please use the following command to install `pygad`:

```bash
conda install -c conda-forge pygad
```

## How to Run

1. Clone this repository to your local machine.

```bash
git clone https://github.com/ma-wenqian/Frame-Structure-Optimization-HKU-IMSE7111
cd Frame-Structure-Optimization-HKU-IMSE7111
```

2. Visualize the frame model:

```bash
python frame_model.py
```

3. Run Bayesian Optimization:

```bash
python bayesian_solution.py --n_draws 50 --n_chains 4 --fitness_type fitness1
python bayesian_solution.py --n_draws 50 --n_chains 4 --fitness_type fitness2
python bayesian_solution.py --n_draws 50 --n_chains 4 --fitness_type fitness3

python bayesian_solution.py --n_draws 100 --n_chains 4 --fitness_type fitness1
python bayesian_solution.py --n_draws 100 --n_chains 4 --fitness_type fitness2
python bayesian_solution.py --n_draws 100 --n_chains 4 --fitness_type fitness3

python bayesian_solution.py --n_draws 150 --n_chains 4 --fitness_type fitness1
python bayesian_solution.py --n_draws 150 --n_chains 4 --fitness_type fitness2
python bayesian_solution.py --n_draws 150 --n_chains 4 --fitness_type fitness3
```

4. Run Genetic Algorithm Optimization:

```bash
python ga_solution.py --num_generations 30 --sol_per_pop 10 --fitness_type fitness1
python ga_solution.py --num_generations 150 --sol_per_pop 100 --fitness_type fitness1
python ga_solution.py --num_generations 30 --sol_per_pop 10 --fitness_type fitness2
python ga_solution.py --num_generations 150 --sol_per_pop 100 --fitness_type fitness2
```

5. Summarize and compare results:

```bash
python Summarize_result.py
```

6. Plot GA fitness history:

```bash
python plot_ga_fitness_history.py
```

## Results and Output

- All optimization results, fitness histories, and summary statistics are saved in the `results/` directory.
- Bayesian optimization results (including posterior plots and summaries) are in `results/Bayesian_results/`.
- Genetic Algorithm results and fitness history plots are in `results/ga_results/`.
- The `result_summary.csv` file provides a summary comparison of different optimization runs.

## Data and Code Availability

All code and data for this project are publicly available on HKU DataHub:

- DOI: xxxxxx
- [HKU DataHub Project Link](https://datahub.hku.hk/)

Please cite this project as follows:

```bibtex
@misc{imse7111_project_2025,
  title = {Optimizing Frame Structure Resilience to Seismic Load Using Bayesian Optimization and Genetic Algorithms},
  author = {Vinkey (Wenqian MA) },
  year = {2025},
  howpublished = {HKU DataHub},
  doi = {xxxxxx},
  url = {https://datahub.hku.hk/}
}
```

## Project Description

This project aims to optimize the resilience of frame structures to seismic loads. The frame model is built using OpenSeesPy, a Python library for structural analysis. Bayesian Optimization and Genetic Algorithms are applied to find the optimal structural parameters that enhance resilience. The workflow includes model definition, seismic analysis, optimization, and result visualization.

## Authors

This project was developed as part of the group project for the HKU course IMSE7111.

**Contributors:**
- Vinkey (Wenqian MA), The University of Hong Kong

For questions or collaboration, please contact me at.

