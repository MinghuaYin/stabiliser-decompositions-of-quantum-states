import math
import time

import cvxpy as cp
import numpy as np
import scipy.sparse as spr


def ham(n: int) -> int:
    weight = 0

    while n:
        weight += 1
        n &= n - 1

    return weight


def optimize_stab_extent_T(n: int, print_output=False):
    # B = np.loadtxt('data/1_qubit_B.csv', dtype=complex, delimiter=',')
    B = spr.load_npz(f'data/{n}_qubit_B.npz')
    num_stab_states, num_non_comp_stab_states = B.shape

    exp_pi_i_over_4 = 1/math.sqrt(2) * (1 + 1j)
    c = np.zeros(num_stab_states, dtype=complex)
    c[:2**n] = 1/2**(n/2) * np.array([exp_pi_i_over_4 ** ham(i)
                                      for i in range(2**n)], dtype=complex)
    c = spr.csc_array(c).reshape((num_stab_states, 1))

    x_l1 = cp.Variable((num_non_comp_stab_states, 1), complex=True)

    obj = cp.Minimize(cp.norm(B @ x_l1 + c, 1)**2)

    prob = cp.Problem(obj)
    solution = prob.solve(solver='GUROBI')  # TODO Is ECOS the best?

    if print_output:
        print(f"status: {prob.status}")

        print(f"optimal objective value: {obj.value}")
        # print(repr(x_l1.value))
        print(f'{(4/(2 + math.sqrt(2)))**n = }')

    return B, obj.value, x_l1.value


if __name__ == '__main__':
    # B_old = np.loadtxt('data/1_qubit_B.csv', dtype=complex, delimiter=',')
    # B = spr.load_npz('data/5_qubit_B.npz')

    n = 4
    start = time.perf_counter()
    B, optimal_val, x = optimize_stab_extent_T(n, True)
    print(f'Time elapsed: {time.perf_counter() - start}')
