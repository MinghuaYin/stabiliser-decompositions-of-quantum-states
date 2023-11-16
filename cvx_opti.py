import math
import pickle
import sys
import time

import cvxpy as cp
import numpy as np
import scipy.sparse as spr

from F2_helper.F2_helper import fast_log2, int_to_array
from functools import reduce
from itertools import combinations, product
from stabiliser_state.Check_Matrix import Check_Matrix
from typing import Tuple


def ham(n: int) -> int:
    weight = 0

    while n:
        weight += 1
        n &= n - 1

    return weight


def T_state(n) -> np.ndarray:
    exp_pi_i_over_4 = 1/math.sqrt(2) * (1 + 1j)
    return 1/2**(n/2) * np.array([exp_pi_i_over_4 ** ham(i)
                                  for i in range(2**n)], dtype=complex)


def dicke_state(n, weight) -> np.ndarray:
    dicke_state = np.zeros(1 << n, dtype=np.int8)
    combos = combinations(range(n), weight)

    for c in combos:
        index_str = ''.join([('1' if i in c else '0') for i in range(n)])
        index = int(index_str, 2)
        dicke_state[index] = 1

    return dicke_state / np.linalg.norm(dicke_state)


def CCZ_state(n) -> np.ndarray:
    return 1/2**(n/2) * np.array([1]*((1 << n)-1) + [-1], dtype=np.int8)


def W_state(n) -> np.ndarray:
    def g(x: Tuple[int]):
        return reduce(lambda a, b: a ^ b, x) + sum(x)

    return 1/2**(n/2) * np.array([np.exp(1j * math.pi * g(x)/4)
                                  for x in product((0, 1), repeat=n)], dtype=complex)


def optimize_stab_extent(state: np.ndarray, n: int, print_output=True, solver='GUROBI') -> \
        Tuple[np.ndarray, float, np.ndarray]:
    B = spr.load_npz(f'data/{n}_qubit_B.npz')
    num_stab_states, num_non_comp_stab_states = B.shape

    c = np.zeros(num_stab_states, dtype=complex)
    c[:(1 << n)] = state
    c = spr.csc_array(c).reshape((num_stab_states, 1))

    x_l1 = cp.Variable((num_non_comp_stab_states, 1), complex=True)

    obj = cp.Minimize(cp.norm(B @ x_l1 + c, 1)**2)
    # obj = cp.Minimize((B @ x_l1 + c).count_nonzero())  # TODO Stabilizer rank lol?

    prob = cp.Problem(obj)
    eps = 1e-10
    if solver == 'GUROBI':
        solution = prob.solve(
            solver='GUROBI', BarQCPConvTol=eps)  # verbose=True
    elif solver == 'SCS':
        solution = prob.solve(solver='SCS', eps=eps)  # verbose=True
    elif solver == 'ECOS':
        solution = prob.solve(solver='ECOS', reltol=eps)

    if print_output:
        print(f"status: {prob.status}")

        print(f"optimal objective value: {obj.value}")
        # print(repr(x_l1.value))
        print(f'{(4/(2 + math.sqrt(2)))**n = }')

    return B, obj.value, x_l1.value


def more_precise_soln(n: int, B: spr.sparray, x: np.ndarray,
                      non_stab_state: np.ndarray, rnd_dec: int = 3) -> Tuple[float, np.ndarray, np.ndarray]:
    with open(f'data/{n}_qubit_subgroups.data', 'rb') as reader:
        xmatrs = pickle.load(reader)

    num_stab_states = B.shape[0]

    c = np.zeros(num_stab_states, dtype=complex)
    c[:(1 << n)] = non_stab_state
    c = spr.csc_array(c).reshape((num_stab_states, 1))

    x_sparse = spr.csc_array(x)

    nz_indices = round(c + B @ x_sparse, rnd_dec).nonzero()[0]
    state_vectors = spr.dok_array((1 << n, len(nz_indices)), dtype=complex)
    for j, index in enumerate(nz_indices):
        basis_num = index // (1 << n)
        signs = int_to_array(index % (1 << n), n)

        sv = Check_Matrix.from_binary_matrix(
            xmatrs[basis_num], signs).get_stabiliser_state().get_state_vector()
        sv = sv / np.linalg.norm(sv)
        state_vectors[:, j] = sv

    soln = np.linalg.lstsq(state_vectors.toarray(),
                           non_stab_state, rcond=None)[0]
    extent = np.linalg.norm(soln, 1)**2
    return extent, state_vectors.toarray(), soln


def combine(state, n, print_output=True, solver='GUROBI', rnd_dec=3):
    """
    Returns:

    extent: float

    state_vectors: np.ndarray

    soln: np.ndarray

    """

    B, optimal_val, x = optimize_stab_extent(state, n, print_output, solver)
    return more_precise_soln(n, B, x, state, rnd_dec)


if __name__ == '__main__':
    n = int(sys.argv[1])
    solver = 'GUROBI' if len(sys.argv) == 2 else sys.argv[2]

    print(f'{n = }')
    start = time.perf_counter()
    B, optimal_val, x = optimize_stab_extent(T_state(n), n, True, solver)
    np.save(f'data/{n}_qubit_soln', x)
    print(f'Time elapsed: {time.perf_counter() - start}')
