#!/usr/bin/env python3

"""
Code that invokes functions from cvx_opti.py to efficiently find stabiliser extent.
"""

import time

import cvx_opti as op
import multiprocessing as mp
import numpy as np

min_n = 6
max_n = 6


class State():
    def __init__(self, name: str, vector: np.ndarray) -> None:
        self.name = name
        self.vector = vector


start = time.perf_counter()

with mp.Pool() as pool:
    async_res = []

    for n in range(min_n, max_n + 1):
        # T = State(f'T^{n}', op.T_state(n))
        H = State(f'H^{n}', op.H_state(n))
        dicke_states = [State(f'D^{n}_{k}', op.dicke_state(n, k))
                        for k in range(1, n)]
        CCZ = State(f'CC^{n - 1}Z', op.CCZ_state(n-1))
        # W = State(f'W_{n}', op.W_state(n))
        # n_qubit_states = [T, CCZ, W] + dicke_states
        n_qubit_states = [H, CCZ] + dicke_states

        async_res += [
            (state,
             pool.apply_async(op.optimize_stab_extent, (state.vector, n),
                              {'print_output': False, 'solver': 'GUROBI',
                               'do_complex': False}))
            for state in n_qubit_states]

    to_print = 'State\t\tStabilizer extent\t\tTime elapsed (s)\n' \
               '------------------------------------------------------------------------------\n'
    for state, r in async_res:
        soln, extent, state_vectors, time_elapsed = r.get()
        extent_str = '---' if extent is None else f'{extent: .8f}'
        to_print += f'{state.name.ljust(8)}\t\t{extent_str.ljust(10)}' \
                    f'\t\t{time_elapsed}\n'
        np.save(f'opti_data/{state.name}_state_vectors', state_vectors)
        np.save(f'opti_data/{state.name}_coeffs', soln)
    print(to_print)

print(f'Time elapsed: {time.perf_counter() - start} s')
