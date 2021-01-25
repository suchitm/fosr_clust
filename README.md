# fosr_clust

This repository contains the relevant materials for reproducing the analysis of the age-specific fertility rate data presented in the the paper [Simultaneous Variable Selection, Clustering, and Smoothing in Function on Scalar Regression](https://arxiv.org/pdf/1906.10286.pdf). 

The repository is organized as follows: 
  - <tt>data</tt>: contains the dataset in the <tt>.Rdata</tt> format
  - <tt>samplers</tt> contains the code needed to run the DP and DPPM models
    * The files <tt>fosr_dp.jl</tt> and <tt>fosr_dppm.jl</tt> contain functions to run the models and return results
  - The script to run both models and create the plots is in <tt>run_models.jl</tt>
  
The main functions to execute the models are `fosr_dp` and `fosr_dppm`. FOSR DP only uses the Dirichlet process prior while DPPM uses the Dirichlet process plus a point mass. Each function as the same inputs, the important ones being: a matrix `Y` which is an N by M matrix of responses, `W`, a matrix of predictors that will not be clustered, `X`, the matrix of predictors which will be clustered, and `R`, a matrix containing the smoothing penalty for the coefficient estimates. The number of iterations `n_iter` should be set relatively high, and half of the iterations will be used for burn in. 
