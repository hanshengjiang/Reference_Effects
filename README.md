# IPRE: Intertemporal Pricing under Reference Effects

## Overview
This repository contains numerical implementation for the paper [Intertemporal Pricing via Nonparametric Estimation: Integrating Reference Effects and Consumer Heterogeneity](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3702824). Along with the reproduction code, this repositiory code also contains general functions for implementing nonparametric estimation of consumer heterogeneity.

<!---These files have been developed and tested in Python version 3.7.4 and R version 3.6.1.-->

## Folders
- `scripts/`: Python and R files
- `illustrations/`: two `.png` pictures illustrating logit demand, and how it depends on reference price and price
- `simulation_results`: estimated coefficients in simulation stored in `.csv` files
- `pricing_output/`: `.png` pictures showing pricing policy and cumulative revenue, for both simulation and MSOM (real) data
- `MSOM_data_cleaned/`: extracted feature data in `.csv` files, ready as inputs of the estimation algorithm
- `MSOM_data_estimated/`: estimated coefficients of MSOM data stored in `.csv` files
- `MSOM_data_optimized/`: revenue comparison for real data study
- `MSOM_Data/`: MSOM-JD.com dataset

## Scripts and Modules
Each python script in `scripts/` starting with `run_` is used for one run of a certain numerical experiment, while each python scipt ending with `_py` defines some functions to be imported by other files. 

Based on the purposes of all the scripts, we further categorize them into the following modules.

- Data preprocessing and feature extraction
  - `run_data_cleaning.py, py_MSOM_cleaning.py`, `run_extract_features.py, run_freq_user.py, run_freq_estimate.py, ` 
- Heterogeneous Reference Effects Estimation
  - Functions: `py_estimation.py, cross_validation.py, mmnl_simualtion.py` 
  - For simulated data: `run_mmnl_estimation_simulation.py`
  - For MSOM data: `run_mmnl_estimation.py`, `run_mmnl_estimation_compare.py`
- Pricing Optimization
  - Functions: `optimal_pricing_policy_exp_update.py`
  - For simulated data: `run_pricing_optimization.py`
  - For MSOM data: `run_mmnl_pricing_optimization.py`, `run_mmnl_revenue_compare.py`


## Real Data and Access
The MSOM-JD.com dataset can be donwloaded from this [link](https://connect.informs.org/msom/events/datadriven2020) given appropariate acess, and general introduction to the dataset is available in this [paper](https://pubsonline.informs.org/doi/abs/10.1287/msom.2020.0900). To be compatible with the codes, the uncompressed `.csv` data files should be stored in the folder `./MSOM_Data/`.
