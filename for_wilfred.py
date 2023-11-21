
import scipy.sparse as spr
import time

n = 6


def main_c(num_of_stab_states):
    B = spr.dok_array((num_of_stab_states, num_of_stab_states - (1 << n)),
                      dtype=complex)
    for i in range(30):
        B_partial = spr.load_npz(f'data/{n}_qubit_B_{i}.npz')
        for k, v in B_partial.items():
            B[k] = v

    return B.tocsc()


if __name__ == '__main__':
    # id = int(sys.argv[1])
    start = time.perf_counter()
    # main_a()
    # B = main_b(id)
    B = main_c(315057600)
    spr.save_npz(f'data/{n}_qubit_B', B)
    print(f'Time elapsed: {time.perf_counter() - start}')
