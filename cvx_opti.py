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


def optimize_stab_extent(state: np.ndarray, n: int, print_output=True, solver='GUROBI') -> \
        Tuple[spr.sparray, float, np.ndarray]:
    """

    Returns
    -------
    B: spr.sparray

    extent: float

    x: np.ndarray

    """

    B = spr.load_npz(f'data/{n}_qubit_B.npz')
    num_stab_states, num_non_comp_stab_states = B.shape

    c = np.zeros(num_stab_states, dtype=complex)
    c[:(1 << n)] = state
    c = spr.csc_array(c).reshape((num_stab_states, 1))

    x_l1 = cp.Variable((num_non_comp_stab_states, 1), complex=True)

    obj = cp.Minimize(cp.norm(B @ x_l1 + c, 1))
    # obj = cp.Minimize((B @ x_l1 + c).count_nonzero())  # TODO Stabilizer rank lol?

    prob = cp.Problem(obj)
    eps = 1e-10

    try:
        if solver == 'GUROBI':
            prob.solve(solver='GUROBI', BarQCPConvTol=eps)  # verbose=True
        elif solver == 'SCS':
            prob.solve(solver='SCS', eps=eps)  # verbose=True
        elif solver == 'ECOS':
            prob.solve(solver='ECOS', reltol=1e-9)
    except cp.SolverError:
        traceback.print_exc()

    if print_output:
        print(f"status: {prob.status}")

        print(f"optimal objective value: {obj.value}")
        print(f'{obj.value**2 = }')
        # print(repr(x_l1.value))
        print(f'{(4/(2 + math.sqrt(2)))**n = }')

    return B, obj.value, x_l1.value


def more_precise_soln(n: int, B: spr.sparray, x: np.ndarray,
                      non_stab_state: np.ndarray, rnd_dec: int = 4) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    with open(f'data/{n}_qubit_subgroups.data', 'rb') as reader:
        xmatrs = pickle.load(reader)

    num_stab_states = B.shape[0]

    c = np.zeros(num_stab_states, dtype=complex)
    c[:(1 << n)] = non_stab_state
    c = spr.csc_array(c).reshape((num_stab_states, 1))

    x_sparse = spr.csc_array(x, dtype=complex)
    old_soln = c + B @ x_sparse

    nz_indices = round(old_soln, rnd_dec).nonzero()[0]
    state_vectors = spr.dok_array((1 << n, len(nz_indices)), dtype=complex)
    for j, index in enumerate(nz_indices):
        basis_num = index // (1 << n)
        signs = int_to_array(index % (1 << n), n)

        sv = Check_Matrix.from_binary_matrix(
            xmatrs[basis_num], signs).get_stabiliser_state().get_state_vector()
        # sv = sv / np.linalg.norm(sv)
        state_vectors[:, j] = sv
    state_vectors = state_vectors.toarray()

    # TODO There's a bug here somewhere... :/
    soln_var = cp.Variable((state_vectors.shape[1], 1), complex=True)
    obj = cp.Minimize(cp.norm(soln_var, 1))
    constrs = [state_vectors @ soln_var == non_stab_state.reshape((1 << n, 1))]
    prob = cp.Problem(obj, constrs)

    prob.solve(solver='ECOS')
    extent = obj.value
    # if extent is None:
    #     pass
    soln = soln_var.value
    return old_soln.toarray(), extent, state_vectors, soln


def combine(state, n, print_output=True, solver='GUROBI', rnd_dec=4):
    """
    Returns:

    extent: float

    state_vectors: np.ndarray

    soln: np.ndarray

    """

    B, optimal_val, x = optimize_stab_extent(state, n, print_output, solver)
    return more_precise_soln(n, B, x, state, rnd_dec)


if __name__ == '__main__':
    # n = int(sys.argv[1])
    # solver = 'GUROBI' if len(sys.argv) == 2 else sys.argv[2]

    start = time.perf_counter()
    optimize_stab_extent(T_state(2), 2, solver='GUROBI')
    print(f'Time elapsed: {time.perf_counter() - start}')

    start = time.perf_counter()
    optimize_stab_extent(T_state(2), 2, solver='ECOS')
    print(f'Time elapsed: {time.perf_counter() - start}')

    # print(f'{n = }')
    # start = time.perf_counter()
    # B, optimal_val, x = optimize_stab_extent(T_state(n), n, True, solver)
    # np.save(f'data/{n}_qubit_soln', x)
    # print(f'Time elapsed: {time.perf_counter() - start}')
