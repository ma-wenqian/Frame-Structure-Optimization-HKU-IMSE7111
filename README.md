# Optimizing Frame Structure Resilience to Seismic Load Using Bayesian Optimization and Genetic Algorithms

This repository contains the code for the group project of the HKU course IMSE7111. The project focuses on optimizing the resilience of frame structures to seismic loads using Bayesian Optimization and Genetic Algorithms.

## Project Structure

- `frame_model.py`: Contains the implementation of the frame model using OpenSeesPy.
- `bayesian_solution.py`: Implements Bayesian Optimization for structural optimization.
- `ga_solution.py`: Implements Genetic Algorithms for structural optimization.
- `elcentro.txt`: Contains seismic load data used in the analysis.


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

## How to Run

1. Clone this repository to your local machine.

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Run the `frame_model.py` script to visualize the frame model:

```bash
python frame_model.py
```

This will display a plot of the frame model structure.

3. Use `bayesian_solution.py` and `ga_solution.py` to perform optimization tasks. For example:

```bash
python bayesian_solution.py
python ga_solution.py
```

4. The `elcentro.txt` file is used as input for seismic load data, and the results of the Genetic Algorithm optimization are saved in `genetic_result.pkl`.

## Project Description

This project aims to optimize the resilience of frame structures to seismic loads. The frame model is built using OpenSeesPy, a Python library for structural analysis. Bayesian Optimization and Genetic Algorithms are applied to find the optimal structural parameters that enhance resilience.

## Authors

This project is developed as part of the group project for the HKU course IMSE7111.