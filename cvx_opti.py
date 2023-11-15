import math
import time

import cvxpy as cp
import numpy as np
import scipy.sparse as spr

sqrt2 = 1.4142135623730950


def ham(n: int) -> int:
    weight = 0

    while n:
        weight += 1
        n &= n - 1

    return weight


def optimize_stab_extent_T(n: int, print_output=False):
    B = spr.load_npz(f'data/{n}_qubit_B.npz')
    num_stab_states, num_non_comp_stab_states = B.shape

    exp_pi_i_over_4 = 1/math.sqrt(2) * (1 + 1j)
    c = np.zeros(num_stab_states, dtype=complex)
    c[:2**n] = 1/2**(n/2) * np.array([exp_pi_i_over_4 ** ham(i)
                                      for i in range(2**n)], dtype=complex)
    c = spr.csc_array(c).reshape((num_stab_states, 1))

    x_l1 = cp.Variable((num_non_comp_stab_states, 1), complex=True)

    obj = cp.Minimize(cp.norm(B @ x_l1 + c, 1)**2)
    # obj = cp.Minimize((B @ x_l1 + c).count_nonzero())  # TODO Stabilizer rank lol?

    prob = cp.Problem(obj)
    solution = prob.solve(solver='GUROBI')  # TODO Is ECOS the best?

    if print_output:
        print(f"status: {prob.status}")

        print(f"optimal objective value: {obj.value}")
        # print(repr(x_l1.value))
        print(f'{(4/(2 + math.sqrt(2)))**n = }')

    return B, obj.value, x_l1.value


def more_precise_soln(x: np.ndarray):
    nonzero_indices = np.nonzero(x)[0]
    # TODO Finish


if __name__ == '__main__':
    n = 4
    start = time.perf_counter()
    B, optimal_val, x = optimize_stab_extent_T(n, True)
    # np.save(f'data/{n}_qubit_soln', x)
    print(f'Time elapsed: {time.perf_counter() - start}')
