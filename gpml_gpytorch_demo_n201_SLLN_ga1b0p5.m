%% Add the gpml path and load it

gpml_path = "./gpml-matlab-v3.6-2015-07-07";
addpath(gpml_path);
startup;
rng("default"); % Default seed
clear; close all; % Prepare plotting environment

%% Import the data

% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 3, "Encoding", "UTF-8");

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["VarName1", "y1", "x"];
opts.VariableTypes = ["double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
simplelinearlownoise = readtable("../MATLAB/data/simple_linear_lownoise_n201.csv", opts);

% Clear temporary variables
clear opts

%% Structure the data

train_y1 = table2array(simplelinearlownoise(:,"y1"));
train_x = table2array(simplelinearlownoise(:,"x"));

%% Set up the Gaussian process regression model

% Model likelihood
likfunc = {@likGauss}; 
sn = 0.1;
hypSEard.lik = log(sn);

% Covariance function and priors for hyperparameters
covfunc = {@covSEard}; % This becomes to covSEiso with a single covariate
ells = ones(size(train_x, 2), 1)./2; % Initialize each length scale to 1/2
sf = 1; % Initialize scale factor to 1/2
% The ARD exponential function has size(X,2)+1 hyperparamters.
hypSEard.cov = log([ells; sf]);

% Mean function
meanfunc = {@meanZero};

% Define hyperpriors
prior.cov = {1:(size(train_x, 2) + 1)};
for i = 1:size(train_x, 2)
    prior.cov{i} = {@priorTransform,@exp,@exp,@log,{@priorGamma,1,1/2}}; % Gamma prior on length scales
end
prior.cov{(size(train_x, 2) + 1)} = {@priorTransform,@exp,@exp,@log,{@priorGamma,1,1/2}}; % Gamma prior on scale factor 
prior.lik = {{@priorTransform,@exp,@exp,@log,{@priorGamma,1,1/2}}}; % Gamma prior on noise

% Define the inference method
inffunc = {@infPrior, @infExact, prior};
p.method = 'LBFGS';
p.length = 100;

% Find the MAP estimates for the model hyperparameters.
hypSEard = minimize_v2(hypSEard, @gp, p, inffunc, meanfunc, covfunc, likfunc, train_x, train_y1);

% Report negative log marginal likelihood of the data with the optimized model hyperparameters.
nlml2 = gp(hypSEard, inffunc, meanfunc, covfunc, likfunc, train_x, train_y1);

% Make predictions
predict_z = linspace(-2, 2, 101)'; % Grid on which to predict values
[m, s2] = gp(hypSEard, inffunc, meanfunc, covfunc, likfunc, train_x, train_y1, predict_z);

%% Learn model parameters with an HMC sampler

% Define parameters for the sampler
num_chains  = 1;
num_samples = 100;
burn_in     = 200;
jitter      = 1e-1;

% Break up hypSEard for some easier algebra
hypSEard_ind = false(size(unwrap(hypSEard)));
hypSEard_ind(1:3) = true;
hypSEard_0 = unwrap(hypSEard);
hypSEard_0 = hypSEard_0(hypSEard_ind);

% Define the function on which to minimize nll
f = @(unwrapped_theta) customize_ll(unwrapped_theta, hypSEard_ind, hypSEard, inffunc, meanfunc, covfunc, train_x, train_y1);  

% Define the sampler
hmc = hmcSampler(f, hypSEard_0 + randn(size(hypSEard_0)) * jitter);

% Tune the sampler
tic;
[hmc, tune_info] = tuneSampler(hmc, 'verbositylevel', 2, 'numprint', 10, 'numstepsizetuningiterations', 100, 'numstepslimit', 500);
toc;

% Train the HMC sampler
tic;
[chain, endpoint, acceptance_ratio] = drawSamples(hmc, 'start', hypSEard_0 + jitter * randn(size(hypSEard_0)), 'burnin', burn_in, 'numsamples', num_samples, 'verbositylevel', 1, 'numprint', 10);
toc;

clear mus;
clear s2s;

for i=1:size(chain,1)
    hypSEard_0 = unwrap(hypSEard);
    hypSEard_0(hypSEard_ind)=chain(i,:);
    hypSEard_0 = rewrap(hypSEard, hypSEard_0);
    [~, ~, mu, s2] = gp(hypSEard_0, inffunc, meanfunc, covfunc, [], train_x, train_y1, predict_z);
    mus{i} = mu;
    s2s{i} = s2;
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

fig = figure(2);
clf;
f = [gmm_mean+1.96*sqrt(gmm_var); flip(gmm_mean-1.96*sqrt(gmm_var),1)];
fill([predict_z; flip(predict_z, 1)], f, [7 7 7]/8, 'edgecolor', 'none');
hold on; plot(predict_z, gmm_mean);
scatter(train_x, train_y1); title("Simple linear DGP with low noise (n=201)");

fig = figure(3);
expchain = exp(chain);
plotmatrix(expchain); title("Posterior hypers: simple linear DGP with low noise (n=201)")


