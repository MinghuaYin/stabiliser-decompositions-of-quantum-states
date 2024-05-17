# Bases for optimising stabiliser decompositions of quantum states

See paper [arXiv:2311.17384](https://arxiv.org/abs/2311.17384) for the mathematical theory.

This repository consists of the following files/directories:

### Python scripts
- **`check_matrices.py`**: incrementally generates all the check matrices (without the column of signs) for
$n$-qubit stabiliser states. It also generates a dictionary of stabiliser state
data in preparation for generating the matrix B of linearly dependent triples.
- **`B_helper_functions.py`**: contains functions that help to generate the $B$ matrix whose columns are a basis of triples for the space $\mathcal{L}_n$ of linear dependencies.
- **`B_gen_slurm.py`**: invokes functions from `B_helper_functions.py` to incrementally generate the B matrix of linearly dependent triples. It was tailored for use on an HPC with the Slurm Workload Manager.
- **`cvx_opti.py`**: contains functions for finding the stabiliser extent of various $n$-qubit states.
- **`opti_run.py`**: invokes functions from `cvx_opti.py` to efficiently find stabiliser extent.

### Output text files
- **`6_qubit_stab_extents.out`**: the computed stabiliser extents of a number of 6-qubit states, as well as timings.
- **`timings.txt`**: timings (in seconds) for running the code for $n = 1, \ldots, 5$ qubits.

### Data directories
- **`data`**: files, for $n = 1, \ldots, 6$ qubits, that contain
    - the matrix $B$ in Python-ready `.npz` format,
    - the check matrices (`..._subgroups_polished.data`) in a readable format for the code,
    - (for $n \leq 3$) a matrix whose columns are the $n$-qubit stabiliser states (`..._matrix_sorted.csv`), in a human-readable format,
    - (for $n \leq 3$) the $B$ matrix in a human-readable `.csv` format.
    
  Note that, for $n = 6$, the data here is restricted to the set of *real* stabiliser states, due to computational complexity. This means that, with this data, we can only accurately compute the stabiliser extent of a real state (since every real state has an extent-optimal decomposition into real stabiliser states, with real coefficients).
- **`opti_data`**: files that contain, for a number of 6-qubit states $\ket{\psi}$,
    - the state vectors of the stabiliser states present in an extent-optimal decomposition of $\ket{\psi}$,
    - the coefficients in this decomposition.

  Both pieces of information are provided in both Python-ready `.npy` format and in human-readable `.csv` format.

---

The remaining directories are derived from [this code](https://github.com/WilfredSalmon/Stabiliser), with a few modifications.