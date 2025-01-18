# HOCMIM

This repository contains the MATLAB implementation of the **High-order Conditional Mutual Information Maximization (HOCMIM)** algorithm for feature selection. 

The code is designed to handle datasets with high-order dependencies effectively and includes a parallelized implementation for efficiency.

## Features
- Selects features based on conditional mutual information (CMI).
- Supports high-order redundancy evaluations.
- Option to use fast approximations based on Markov model assumptions.
- Verbose output for tracking progress.

## Installation
Clone this repository and ensure MATLAB is installed with the required toolbox (`MIToolboxMex`).

## Usage

### Function Signature
```matlab
[S, tElapsed, CMIapp, par, Order] = pHOCMIMauto(X, Y, K, n, fast, verbose)
```

### Example usage
```matlab
% Example usage
X = rand(100, 20); % 100 samples, 20 features
Y = randi([0, 1], 100, 1); % Binary labels
K = 5; % Select 5 features
n = 3; % Third-order approximation
[S, tElapsed, CMIapp, par, Order] = pHOCMIMauto(X, Y, K, n, true, true);
disp(S);
```


