#!/usr/bin/env python3

"""
Helper functions for finding the stabiliser extent of various n-qubit states.
"""

import math
import pickle
import sys
import time
import traceback

import cvxpy as cp
import numpy as np
import scipy.sparse as spr

from F2_helper.F2_helper import fast_log2, int_to_array
from functools import reduce
from itertools import combinations, product
from stabiliser_state.Check_Matrix import Check_Matrix
from typing import Tuple

sqrt2 = 1.4142135623730950


def ham(n: int) -> int:
    weight = 0

    while n:
        weight += 1
        n &= n - 1

    return weight


def T_state(n) -> np.ndarray:
    exp_pi_i_over_4 = 1/math.sqrt(2) * (1 + 1j)
    return 1/sqrt2**n * np.array([exp_pi_i_over_4 ** ham(i)
                                  for i in range(1 << n)], dtype=complex)


def H_state(n) -> np.ndarray:
    cos_pi_over_8 = math.sqrt(2 + sqrt2)/2
    sin_pi_over_8 = math.sqrt(2 - sqrt2)/2
    return np.array([cos_pi_over_8**(n-ham(i)) * sin_pi_over_8**ham(i)
                     for i in range(1 << n)])


def dicke_state(n, weight) -> np.ndarray:
    dicke_state = np.array(
        [1 if ham(i) == weight else 0 for i in range(1 << n)])
    return dicke_state / np.linalg.norm(dicke_state)


def CCZ_state(n_minus_1) -> np.ndarray:
    n = n_minus_1 + 1
    return 1/sqrt2**n * np.array([1]*((1 << n)-1) + [-1])


def W_state(n) -> np.ndarray:
    def g(x: Tuple[int]):
        return reduce(lambda a, b: a ^ b, x) + sum(x)

    return 1/sqrt2**n * np.array([np.exp(1j * math.pi * g(x)/4)
                                  for x in product((0, 1), repeat=n)], dtype=complex)


def optimize_stab_extent(state: np.ndarray, n: int, print_output=True,
                         solver='GUROBI', rnd_dec: int = 4, do_complex=True) -> Tuple[spr.sparray, float, np.ndarray, float]:
    """

    Returns
    -------
    soln: spr.sparray

    extent: float

    state_vectors: np.ndarray

    time_elapsed: float

    """

    start = time.perf_counter()

    filename = f'data/{n}_qubit_B.npz' if do_complex \
        else f'data/{n}_qubit_B_real.npz'
    B = spr.load_npz(filename)
    num_stab_states, num_non_comp_stab_states = B.shape

    c = np.zeros(num_stab_states, dtype=(complex if do_complex else float))
    c[:(1 << n)] = state
    c = spr.csc_array(c).reshape((num_stab_states, 1))

    x_l1 = cp.Variable((num_non_comp_stab_states, 1), complex=do_complex)

    obj = cp.Minimize(cp.norm(B @ x_l1 + c, 1))

    prob = cp.Problem(obj)
    eps = 1e-10

    try:
        if solver == 'GUROBI':
            prob.solve(solver='GUROBI', BarQCPConvTol=eps,
                       verbose=print_output)
        elif solver == 'SCS':
            prob.solve(solver='SCS', eps=eps, verbose=print_output)
        elif solver == 'ECOS':
            prob.solve(solver='ECOS', verbose=print_output)
    except cp.SolverError:
        print(f'****Oh dear...****\n{state = }')
        traceback.print_exc()
        print('********')

    if print_output:
        print(f"status: {prob.status}")

        print(f"optimal objective value: {obj.value}")
        print(f'{obj.value**2 = }')

    x = x_l1.value

    # Get state vectors for the stabiliser states in the optimal decomposition
    xmatrs = []
    with open(f'data/{n}_qubit_subgroups_polished{"" if do_complex else "_real"}.data', 'rb') \
            as reader:
        try:
            while True:
                xmatrs.extend(pickle.load(reader))
        except EOFError:
            pass

    x_sparse = spr.csc_array(x, dtype=(complex if do_complex else float))
    soln = c + B @ x_sparse

    nz_indices = round(soln, rnd_dec).nonzero()[0]
    state_vectors = spr.dok_array(
        (1 << n, len(nz_indices)), dtype=(complex if do_complex else float))
    for j, index in enumerate(nz_indices):
        basis_num = index // (1 << n)
        signs = int_to_array(index % (1 << n), n)

        sv = Check_Matrix.from_binary_matrix(
            xmatrs[basis_num], signs).get_stabiliser_state().get_state_vector()
        # sv = sv / np.linalg.norm(sv)
        state_vectors[:, j] = sv
    state_vectors = state_vectors.toarray()

    return soln, obj.value**2, state_vectors, time.perf_counter() - start
