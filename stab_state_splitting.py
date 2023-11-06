#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copied from old project --- see helper_functions for new code
import pickle
import time
import numpy as np
from scipy.linalg import null_space

from stabiliser_states import state_from_subgroup


def sort_by_support(n, states: list):
    """
    Sorts a list of stabiliser states in ascending order of support size, and 
    also creates a matrix of all sorted stabiliser states.

    Returns
    -------
    None.

    """

    # Keep each basis intact
    bases = list(enumerate(states[::2**n]))
    bases.sort(key=lambda x: np.count_nonzero(x[1]['vector']))

    states_sorted = []
    for num, b in bases:
        states_sorted += states[2**n * num: 2**n * (num + 1)]

    # print(len(states))
    stab_matrix_sorted = np.hstack([state['vector']
                                    for state in states_sorted])
    np.savetxt(f'data/{n}_qubit_matrix_sorted.csv',
               stab_matrix_sorted, delimiter=',', fmt='%.5e')
    return states_sorted


def split_state(n, j, stab_states: list):
    state = stab_states[j + 2**n]

    phases = [(1 if gen[0] == '-' else 0)
              for gen in state['generator_strs']]

    og_check_mat = state['check_matrix']
    k = list(og_check_mat[0, :]).index(1)
    new_row = np.array([[(0 if i != n + k else 1) for i in range(2*n)]])

    split_check_mat = np.vstack((new_row, og_check_mat[1:, :]))

    # Find two split states
    split_state_1 = state_from_subgroup((n, split_check_mat,
                                         [0] + phases[1:]))
    split_state_2 = state_from_subgroup((n, split_check_mat,
                                         [1] + phases[1:]))

    # Find coefficients in linear dependency
    triple = np.hstack(
        (split_state_1['vector'], split_state_2['vector'], state['vector']))
    coeffs = null_space(triple)
    # Normalise and round
    coeffs = np.around(coeffs / coeffs[2], decimals=8)[:, 0]

    # Find index of each split state
    index_1, index_2 = -1, -1
    for i, test_state in enumerate(stab_states):
        if np.all(np.around(test_state['vector'] - split_state_1['vector'],
                            decimals=3) == 0):
            index_1 = i
        elif np.all(np.around(test_state['vector'] - split_state_2['vector'],
                              decimals=3) == 0):
            index_2 = i

        if index_1 > -1 and index_2 > -1:
            break

    col = np.zeros((len(stab_states), 1), dtype=complex)
    col[index_1, 0] = coeffs[0]
    col[index_2, 0] = coeffs[1]
    col[j + 2**n, 0] = 1

    return col


def get_lin_dep_basis(n):
    start_time = time.perf_counter()

    stab_states = []
    try:
        with open(f'data/{n}_qubit_stab_states.data', 'rb') as reader:
            while True:
                stab_states += pickle.load(reader)
    except EOFError:
        pass

    # Sort the stab states in order of support size
    stab_states = sort_by_support(n, stab_states)

    # Initialise basis matrix
    basis = np.zeros((len(stab_states), len(stab_states) - 2**n),
                     dtype=complex)

    for j in range(len(stab_states) - 2**n):
        # Update appropriate column of basis matrix
        col = np.reshape(split_state(n, j, stab_states), (len(stab_states),))
        basis[:, j] = col

    print(f'Took {time.perf_counter() - start_time} s to split all {n}-qubit '
          'stab states.')

    np.savetxt(f'data/lin_dep_sets/{n}_qubit_nullspace_nice.csv', basis,
               delimiter=',', fmt='%.5e')

    return basis


def get_relative_phases(n):
    basis = np.loadtxt(f'data/lin_dep_sets/{n}_qubit_nullspace_nice.csv',
                       dtype=complex, delimiter=',')
    num_stabs = basis.shape[0]
    rel_phases = [[None for _ in range(num_stabs)] for _ in range(num_stabs)]

    for j, col in enumerate(basis.T):
        non_zero_entries = list(np.nonzero(col)[0])
        non_zero_entries.remove(j + 2**n)
        # if len(non_zero_entries) != 2:
        #     raise RuntimeError('Uh-oh, spaghetti-o')

        a, b = non_zero_entries
        if rel_phases[a][b] is None:
            rel_phases[a][b] = [col[b] / col[a]]
        else:
            rel_phases[a][b].append(col[b] / col[a])

    return rel_phases


if __name__ == '__main__':
    basis = get_lin_dep_basis(3)
