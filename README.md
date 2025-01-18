# HOCMIM

This repository contains the MATLAB implementation of the **High-order Conditional Mutual Information Maximization (HOCMIM)** algorithm for feature selection. The code is designed to handle datasets with high-order dependencies effectively and includes a parallelized implementation for efficiency.

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

### Arguments
- **X**: Training data (matrix of size M x N).
  - `M`: Number of samples.
  - `N`: Number of features.
- **Y**: Target labels (vector of size M x 1).
- **K**: Number of features to select.
- **n**: Order of HOCMIM approximation.
- **fast**: Boolean, use fast approximation (default: `false`).
- **verbose**: Boolean, display verbose output (default: `false`).

### Outputs
- **S**: Selected features.
- **tElapsed**: Time taken for execution.
- **CMIapp**: CMI approximation for each selected feature.
- **par**: Struct containing additional output parameters.
- **Order**: Selection order of the features.

### Example
```matlab
% Example usage
X = rand(100, 20); % 100 samples, 20 features
Y = randi([0, 1], 100, 1); % Binary labels
K = 5; % Select 5 features
n = 3; % Third-order approximation
[S, tElapsed, CMIapp, par, Order] = HOCMIM(X, Y, K, n, true, true);
disp(S);
```

## Citation
If you use this code, please cite the following paper:

Francisco Souza, Cristiano Premebida, Rui Araújo, "High-order conditional mutual information maximization for dealing with high-order dependencies in feature selection," *Pattern Recognition*, Volume 131, 2022.

### BibTeX
```bibtex
@article{Souza2022,
  title={High-order conditional mutual information maximization for dealing with high-order dependencies in feature selection},
  author={Francisco Souza and Cristiano Premebida and Rui Araújo},
  journal={Pattern Recognition},
  volume={131},
  year={2022},
  doi={10.1016/j.patcog.2022.108933}
}
```

## References
1. Francisco Souza, Cristiano Premebida, Rui Araújo, "High-order conditional mutual information maximization for dealing with high-order dependencies in feature selection," *Pattern Recognition*, Volume 131, 2022.

## License
This project is licensed under the MIT License. 
