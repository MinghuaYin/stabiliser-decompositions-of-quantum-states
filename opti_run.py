import time

import cvx_opti as op
import multiprocessing as mp
import numpy as np

max_n = 4
# max_n = 6


class State():
    def __init__(self, name: str, vector: np.ndarray) -> None:
        self.name = name
        self.vector = vector


start = time.perf_counter()

with mp.Pool() as pool:
    async_res = []

    for n in range(1, max_n + 1):
        T = State(f'T^{n}', op.T_state(n))
        dicke_states = [State(f'D_{n}^{k}', op.dicke_state(n, k))
                        for k in range(1, n)]
        CCZ = State(f'CC^{n - 1}Z', op.CCZ_state(n-1))
        W = State(f'W_{n}', op.W_state(n))
        # n_qubit_states = [T, CCZ, W] + dicke_states
        n_qubit_states = [CCZ] + dicke_states

        async_res += [
            (state,
             pool.apply_async(op.combine, (state.vector, n),
                              {'print_output': False, 'solver': 'GUROBI',
                               'rnd_dec': 4, 'do_complex': False}))
            for state in n_qubit_states]

    to_print = 'State\t\t(||old_soln||_1)^2\t\tStabilizer extent squared\t\tTime elapsed\n' \
               '-----------------------------------------------------------------------------------------------------\n'
    for state, r in async_res:
        old_soln, extent, state_vectors, soln, time_elapsed = r.get()
        extent_str = '---' if extent is None else f'{extent**2: .8f}'
        to_print += f'{state.name.ljust(8)}\t\t{np.linalg.norm(old_soln, 1)**2: .8f}' \
                    f'\t\t{extent_str.ljust(10)}\t\t\t\t{time_elapsed}\n'
        # np.save(f'opti_data/{state.name}_state_vectors', state_vectors)
    print(to_print)

print(f'Time elapsed: {time.perf_counter() - start}')
