# fosr_clust

This repository contains the relevant materials for reproducing the analysis of the age-specific fertility rate data presented in the the paper [Simultaneous Variable Selection, Clustering, and Smoothing in Function on Scalar Regression](https://arxiv.org/pdf/1906.10286.pdf). 

The repository is organized as follows: 
  - <tt>data</tt>: contains the dataset in the <tt>.Rdata</tt> format
  - <tt>samplers</tt> contains the code needed to run the DP and DPPM models
    * The files <tt>fosr_dp.jl</tt> and <tt>fosr_dppm.jl</tt> contain functions to run the models and return results
  - The script to run both models and create the plots is in <tt>run_models.jl</tt>
