# gpml_gpytorch_comparison
Compare simple models that fail to train in GPyTorch with their implementation in GPML

The purpose of this exercise is to demonstrate that many GPyTorch models will not train while GPML handles the training data fine. This folder contains scripts needed to generate simple data generating processes that GPyTorch cannot train. I generate four sets of training data using PyTorch, normalize the training data, and save the data in a .csv format. 


Generating data:  x_i ~ i.i.d. N(0,1), u_i ~ i.i.d. N(0,1), n = 101 and n = 201


A simple linear DGP with low noise (SLLN):        y_1i = 1 - x_i + (0.1)*u_i

A simple linear DGP with high noise (SLHN):      y_2i = 1 - x_i + u_i

A simple cubic DGP with low noise (SCLN):         y_3i = 1 - x_i + x^2_i - x^3_i + (0.1)*u_i

A simple cubic DGP with high noise (SCHN):        y_4i = 1 - x_i + x^2_i - x^3_i + u_i


I use the data to train a Gaussian process regression model with zero mean and an isotropic squared exponential kernel using both GPyTorch and GPML. Model parameters are learned in GPyTorch with a "No U-Turn" sampler (NUTS) since HMC sampling with Pyro seems to be broken (see HMC (n=101).ipynb and HMC with Reshape  (n=101).ipynb for examples). Model parameters in GPML are learned with HMC. Each sampler only trains with 100 burn-in with 200 subsequent samples to demonstrate failure. GPyTorch models test ten different gamma priors for each data set while GPML only tests a single prior for each data set. For each failed training instance in GPyTorch, I apply the same training data and the same gamma priors in GPML and find that the model trains with GPML just fine.

The following five of forty gamma priors did not train for GPyTorch in the notebook NUTS with Reshape (n=101).ipynb:

Simple linear DGP w/ low noise:     Gamma(2,2), Gamma(3,1/2)

Simple linear DGP w/ high noise:    

Simple cubic DGP w/ low noise:      Gamma(1,1/2)

Simple cubic DGP w/ high noise:     Gamma(1,1/2), Gamma(1,3/2)


The following eleven of forty gamma priors did not train for GPyTorch in the notebook NUTS with Reshape (n=201).ipynb.


Simple linear DGP w/ low noise:     Gamma(1,1/2), Gamma(1,2), Gamma(2,2), Gamma(3,1), Gamma(3,1/2)

Simple linear DGP w/ high noise:    Gamma(1,1/5)

Simple cubic DGP w/ low noise:      Gamma(2,3/2)

Simple cubic DGP w/ high noise:     Gamma(1,1), Gamma(1,2), Gamma(2,3/2), Gamma(2,2)


GPML scripts will run if demo files and a 'data' folder containing the .csv files are placed in the home MATLAB directory. Compare these GPML outputs to the contents of NUTS with Reshape (n=101).ipynb and NUTS with Reshape (n=201).ipynb. 


The naming convention to find the corresponding successful GPML implementations of the above GPyTorch failures is as follows. All GPML implementations were successful.


gpml_gpytorch_demo_n101_SLLN_ga2b2        (GPML success)

gpml_gpytorch_demo_n101_SLLN_ga3b0p5      (GPML success)

gpml_gpytorch_demo_n101_SCLN_ga1b0p5      (GPML success)

gpml_gpytorch_demo_n101_SCHN_ga1b0p5      (GPML success)

gpml_gpytorch_demo_n101_SCHN_ga1b1p5      (GPML success)

gpml_gpytorch_demo_n201_SLLN_ga1b0p5      (GPML success)

gpml_gpytorch_demo_n201_SLLN_ga1b2        (GPML success)

gpml_gpytorch_demo_n201_SLLN_ga2b2        (GPML success)

gpml_gpytorch_demo_n201_SLLN_ga3b1        (GPML success)

gpml_gpytorch_demo_n201_SLLN_ga3b0p5      (GPML success)

gpml_gpytorch_demo_n201_SLHN_ga1b0p2      (GPML success)

gpml_gpytorch_demo_n201_SCLN_ga2b1p5      (GPML success)

gpml_gpytorch_demo_n201_SCHN_ga1b1        (GPML success)

gpml_gpytorch_demo_n201_SCHN_ga1b2        (GPML success)

gpml_gpytorch_demo_n201_SCHN_ga2b1p5      (GPML success)

gpml_gpytorch_demo_n201_SCHN_ga2b2        (GPML success)


Note: "Reshape" in the ipynb title refers to reshaping PyTorch tensors to speed up implementation. Implementation speeds up because torch linear algebra takes up a lot of resources discarding degenerate tensor dimensions on the fly. Thanks to Yehu Chen for helping me figure this out.









