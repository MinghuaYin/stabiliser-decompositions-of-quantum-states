import cvx_opti as op
import multiprocessing as mp
import numpy as np

max_n = 4
# max_n = 6


class State():
    def __init__(self, name: str, vector: np.ndarray) -> None:
        self.name = name
        self.vector = vector


with mp.Pool() as pool:
    async_res = []

    for n in range(1, max_n + 1):
        T = State(f'T^{n}', op.T_state(n))
        dicke_states = [State(f'D_{n}^{k}', op.dicke_state(n, k))
                        for k in range(1, n+1)]
        CCZ = State(f'CC^{n - 1}Z', op.CCZ_state(n-1))
        W = State(f'W_{n}', op.W_state(n))
        n_qubit_states = [T, CCZ, W] + dicke_states

        async_res += [
            (state,
             pool.apply_async(op.combine, (state.vector, n),
                              {'print_output': False, 'solver': 'ECOS', 'rnd_dec': 4}))
            for state in n_qubit_states]

    print('State\t\tStabilizer extent')
    print('---------------------------------')
    for state, r in async_res:
        extent, state_vectors, soln = r.get()
        print(f'{state.name.ljust(8)}\t{extent: .8f}')
        np.save(f'opti_data/{state.name}_state_vectors', state_vectors)
