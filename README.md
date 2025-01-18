# HOCMIM

This repository contains implementations of the **High-order Conditional Mutual Information Maximization (HOCMIM)** algorithm for feature selection in both MATLAB and Python. The code is designed to handle datasets with high-order dependencies effectively and includes a parallelized implementation for efficiency.

In the original paper, the mutual information implementation uses the regularized approach from [2], which provides better accuracy due to improved estimation in high-dimensional scenarios. You can follow the authors' repository to enable their estimator in HOCMIM. The link is available in the reference section. However, the MIToolbox and default scikit works relativelly well for 5 dimensions, and a `high´ number of examples (>1000).


## Features
- Selects features based on conditional mutual information (CMI).
- Supports high-order redundancy evaluations.

## Installation
### MATLAB
Clone this repository and ensure MATLAB is installed with the required toolbox (`MIToolboxMex`).

### Python
Clone this repository and ensure Python 3.x is installed along with the required packages:
```bash
pip install numpy scikit-learn
```

## Usage

### MATLAB Function Signature
```matlab
[S, tElapsed, CMIapp, par, Order] = HOCMIM(X, Y, K, n, fast, verbose)
```

#### Arguments
- **X**: Training data (matrix of size M x N).
  - `M`: Number of samples.
  - `N`: Number of features.
- **Y**: Target labels (vector of size M x 1).
- **K**: Number of features to select.
- **n**: Max. order of HOCMIM approximation.
- **fast**: Boolean, use fast approximation (default: `false`).
- **verbose**: Boolean, display verbose output (default: `false`).

#### Outputs
- **S**: Selected features.
- **tElapsed**: Time taken for execution.
- **CMIapp**: CMI approximation for each selected feature.
- **par**: Struct containing additional output parameters.
- **Order**: Selection order of the features.

#### MATLAB Example
```matlab
% Example usage
X = rand(100, 20); % 100 samples, 20 features
Y = randi([0, 1], 100, 1); % Binary labels
K = 5; % Select 5 features
n = 3; % Third-order approximation 
[S, tElapsed, CMIapp, par, Order] = HOCMIM(X, Y, K, n, true, true);
disp(S);
```

### Python Function Signature
```python
selected, elapsed_time, cmi_app, order = hocmim(X, Y, K, n, verbose=False)
```

#### Arguments
- **X**: Training data (NumPy array of shape (M, N)).
  - `M`: Number of samples.
  - `N`: Number of features.
- **Y**: Target labels (NumPy array of size M).
- **K**: Number of features to select.
- **n**: Order of HOCMIM approximation.
- **verbose**: Boolean, display verbose output (default: `False`).

#### Outputs
- **selected**: List of selected feature indices.
- **elapsed_time**: Time taken for execution.
- **cmi_app**: CMI approximation for each selected feature.
- **order**: Selection order of the features.

#### Python Example
```python
import numpy as np
from HOCMIM import hocmim

# Example usage
np.random.seed(42)
X = np.random.rand(100, 20)  # 100 samples, 20 features
Y = np.random.randint(0, 2, 100)  # Binary labels
K = 5  # Select 5 features
n = 3  # Third-order approximation

selected_features, elapsed_time, cmi_app, order = hocmim(X, Y, K, n, verbose=True)

print("Selected features:", selected_features)
print("Elapsed time:", elapsed_time)
```

## Citation
If you use this code, please cite the following paper:

Francisco Souza, Cristiano Premebida, Rui Araújo, "High-order conditional mutual information maximization for dealing with high-order dependencies in feature selection," *Pattern Recognition*, Volume 131, 2022. https://www.sciencedirect.com/science/article/pii/S0031320322003764

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
2. K. Sechidis et al., "Efficient feature selection using shrinkage estimators," *Machine Learning Journal*, 2019. github: https://github.com/sechidis/2019-MLJ-Efficient-feature-selection-using-shrinkage-estimators

## License
This project is licensed under the MIT License.
