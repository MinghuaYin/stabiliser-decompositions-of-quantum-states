# Copied from old project

"""
This module contains functions for generating data relating to subgroups
of the n-qubit Pauli group.

In writing this code, we use the fact that, ignoring the phase, there's a
bijection between elements of the Pauli group and bit strings of length
2n (i.e. integers between 0 and 2^(2n) - 1), where the bit string
corresponds to the element's check vector. Hence we represent an
element of the Pauli group (ignoring the phase) as a vector in F_2^(2n).
"""

import numba
import pickle
import time
import itertools as it
import multiprocessing as mp
import numpy as np

from typing import Iterator, Sequence


def np_block(X):
    xtmp1 = np.hstack(X[0])
    xtmp2 = np.hstack(X[1])
    return np.vstack((xtmp1, xtmp2))


# ***** EDIT THIS BEFORE RUNNING *****
n = 6

O = np.zeros((n, n), dtype=np.int8)
I = np.eye(n, dtype=np.int8)
Lambda = np_block(((O, I), (-I, O)))


# TODO Rewrite for greater efficiency
# @numba.jit(nopython=False, parallel=False, forceobj=True)
# def get_rref_matrices(num_rows, rows=None):
#     """
#     Generator that finds all the n by 2n reduced row echelon form matrices
#     (with no zero rows) with the specified number of rows.

#     Note that, by construction, the rows of these matrices are
#     linearly independent.

#     Parameters
#     ----------
#     num_rows : int
#         The number of rows that the rref matrices are to have.
#     rows : numpy.ndarray, optional
#         Any rows (in rref) that have already been generated.

#     Yields
#     ------
#     rref_mat : numpy.ndarray
#         The next rref matrix.

#     """

#     if rows is None:
#         current_num_rows, num_cols = (0, 2*num_rows)
#     else:
#         current_num_rows, num_cols = rows.shape
#     if current_num_rows == num_rows:
#         yield rows
#     # if current_num_rows > num_rows:
#     #     raise ValueError

#     rref_mats_temp = []

#     # Find the index of the column containing the last row's leading 1
#     # list(rows[-1, :]).index(1)
#     if rows is None:
#         start_col = -1
#     else:
#         start_col = np.argmax(rows[-1, :])

#     # Consider all prototype extra rows
#     for extra_row_numeric in range(1, 1 << (num_cols - start_col - 1)):
#         extra_row = list(
#             format(extra_row_numeric, f'0{num_cols - start_col - 1}b'))

#         extra_row = np.array([0 for _ in range(start_col + 1)]
#                               + extra_row, dtype=np.int8)

#         if rows is None:
#             rref_mats_temp.append(extra_row.reshape((1, num_cols)))
#         else:
#             # Check if this extra row's leading 1 is in a column
#             # without any other 1s; if so, add it to the matrix!
#             leading_col = np.argmax(extra_row)
#             if np.sum(rows[:, leading_col]) == 0:
#                 rref_mats_temp.append(
#                     np.vstack((rows, extra_row.reshape((1, num_cols)))))

#     # Recursively run the function until enough rows have been added
#     if current_num_rows == num_rows - 1:
#         for mat in rref_mats_temp:
#             yield mat
#     elif current_num_rows < num_rows - 1:
#         for new_rows in rref_mats_temp:
#             for mat in get_rref_matrices(num_rows, new_rows):
#                 yield mat


def powerset(iterable) -> Iterator:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))


def get_rref_matrices(leading_one_positions: Sequence[int]) -> Iterator[np.ndarray]:
    template = np.zeros((n, 2*n), dtype=np.int8)

    valid_positions = []

    for i, pos in enumerate(leading_one_positions):
        template[i, pos] = 1
        vps = set(range(pos + 1, 2*n)).difference(leading_one_positions)
        valid_positions.append(vps)

    all_combinations = (powerset(vps) for vps in valid_positions)
    for choice in it.product(*all_combinations):
        matrix = np.copy(template)
        for i, positions in enumerate(choice):
            matrix[i, positions] = 1
        yield matrix


def dot_py(A, B):
    m, n = A.shape
    p = B.shape[1]

    C = np.zeros((m, p), dtype=np.int8)

    for i in range(0, m):
        for j in range(0, p):
            for k in range(0, n):
                C[i, j] += A[i, k]*B[k, j]
    return C


dot_nb = numba.jit(numba.int8[:, :](
    numba.int8[:, :], numba.int8[:, :]), nopython=True)(dot_py)


def check_commute(check_matrix):
    """
    Checks whether the generators, given by the check matrix
    check_matrix, commute with each other.

    We use the fact that the generators commute if and only if

    .. math::
        G \Lambda G^T = 0 \pmod{2},

    where :math:`G` is the matrix whose rows are check vectors
    (i.e. check_matrix) and

    .. math::
        \Lambda = \\begin{pmatrix} 0 & I \\\ I & 0 \end{pmatrix}.

    Parameters
    ----------
    check_matrix : np.ndarray

    Returns
    -------
    commute : bool
        Whether or not the generators commute.

    """

    intermediate = dot_nb(check_matrix, Lambda)
    prod = dot_nb(intermediate, check_matrix.T)

    return np.array_equiv(prod % 2, O)


def get_check_matrices(leading_one_positions):
    """

    Parameters
    ----------
    leading_one_positions : Sequence[int]

    Returns
    -------
    None
        If the check matrix is not valid.
    check_matrix : np.ndarray
        check_matrix if it is valid.

    """

    print(f'Begin looking with {leading_one_positions = }')

    good_ones = []

    for index, mat in enumerate(get_rref_matrices(leading_one_positions)):
        # if index % 1_000_000 == 0:
        #     print(f'Testing\n{mat}')

        if check_commute(mat):
            good_ones.append(mat)

    print(f'Finish looking with {leading_one_positions = }')

    return good_ones


# The main function
def get_max_abelian_subgroups():
    """
    Finds the maximal abelian subgroups of the n-qubit Pauli group,
    ignoring the phase of each generator.

    Parameters
    ----------
    n : int

    Returns
    -------
    subgroups : list
        A list containing the check matrices relating to all the maximal,
        abelian subgroups.

    """

    start_time = time.perf_counter()

    subgroups = []

    with mp.Pool() as pool, \
            open(f'data/{n}_qubit_subgroups_a.data', 'ab') as writer:
        results = pool.imap(
            get_check_matrices,
            it.combinations(range(2*n), n),
            chunksize=10
        )

        # results = map(
        #     get_check_matrices,
        #     it.combinations(range(2*n), n)
        # )

        for sublist in results:
            subgroups += sublist
            if len(subgroups) > 100_000:
                pickle.dump(subgroups, writer)
                subgroups = []

        pickle.dump(subgroups, writer)

    print(f'Total elapsed time: {time.perf_counter() - start_time}')

    return subgroups


if __name__ == '__main__':
    subgroups = get_max_abelian_subgroups()

    # To compare
    with open(f'data/{n}_qubit_subgroups.data', 'rb') as reader:
        old_list = pickle.load(reader)
