"""
This module contains functions for generating data relating to subgroups
of the n-qubit Pauli group.

In writing this code, we use the fact that, ignoring the phase, there's a
bijection between elements of the Pauli group and bit strings of length
2n (i.e. integers between 0 and 2^(2n) - 1), where the bit string
corresponds to the element's check vector. Hence we represent an
element of the Pauli group (ignoring the phase) as a vector in F_2^(2n).
"""

import ctypes
import numba
import pickle
import time
import itertools as it
import multiprocessing as mp
import numpy as np

from stabilizer_states import state_from_subgroup

O_shared = None
Lambda_shared = None


def np_block(X):
    xtmp1 = np.hstack(X[0])
    xtmp2 = np.hstack(X[1])
    return np.vstack((xtmp1, xtmp2))


def binary_matrix_rank(A):
    """
    Finds the rank of the binary matrix A by effectively converting it
    into row echelon form.

    Parameters
    ----------
    A : numpy.ndarray

    Returns
    -------
    rank : int
        The rank of matrix A.

    """

    num_rows, num_cols = A.shape
    rank = 0

    for j in range(num_cols):
        # Find the number of rows that have a 1 in the jth column
        rows = []
        for i in range(num_rows):
            if A[i, j] == 1:
                rows.append(i)

        # If the jth column has more than one 1, use row addition to
        # remove all 1s except the first one, then remove the first
        # such row and increase the rank by 1
        if len(rows) >= 1:
            for c in range(1, len(rows)):
                A[rows[c], :] = (A[rows[c], :] + A[rows[0], :]) % 2

            A = np.delete(A, rows[0], 0)
            num_rows -= 1
            rank += 1

    # For each remaining non-zero row in A, increase the rank by 1
    for row in A:
        if sum(row) > 0:
            rank += 1

    return rank


# TODO Can we get the rank more efficiently?
# mat = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
#                 [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
#                 [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#                 [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
#                 [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]], dtype=np.int8)
# parallel speeds up computation only over very large matrices
# @numba.jit(nopython=True, parallel=False)
def gf2elim(M):
    """
    N.B. This method *changes* the matrix M!

    """

    m, n = M.shape
    i = 0
    j = 0
    # rank = 0

    while i < m and j < n:
        # Find the row with the next leading 1
        k = np.argmax(M[i:, j]) + i
        temp = np.copy(M[k])
        # Move this row into position via a swap
        M[k] = M[i]
        M[i] = temp
        aijn = M[i, j:]
        # make a copy otherwise M will be directly affected
        col = np.copy(M[:, j])
        col[i] = 0  # avoid xoring pivot row with itself
        flip = np.outer(col, aijn)
        M[:, j:] = M[:, j:] ^ flip
        i += 1
        j += 1
        # rank += 1

    return M


def get_rref_matrices(num_rows, rows=None):
    """
    Generator that finds all the n by 2n reduced row echelon form matrices
    (with no zero rows) with the specified number of rows.

    Note that, by construction, the rows of these matrices are
    linearly independent.

    Parameters
    ----------
    num_rows : int
        The number of rows that the rref matrices are to have.
    rows : numpy.ndarray, optional
        Any rows (in rref) that have already been generated.

    Yields
    ------
    rref_mat : numpy.ndarray
        The next rref matrix.

    """

    current_num_rows, num_cols = (0, 2*num_rows) \
        if rows is None else rows.shape
    if current_num_rows == num_rows:
        yield rows
    # if current_num_rows > num_rows:
    #     raise ValueError

    rref_mats_temp = []

    # Find the index of the column containing the last row's leading 1
    start_col = -1 if rows is None else list(rows[-1, :]).index(1)

    # Consider all prototype extra rows
    for extra_row in \
        it.islice(it.product((0, 1), repeat=num_cols - start_col - 1),
                  1, None):

        extra_row = np.array(tuple(0 for _ in range(start_col + 1))
                             + extra_row, dtype=np.int8)

        if rows is None:
            rref_mats_temp.append(extra_row.reshape(1, num_cols))
        else:
            # Check if this extra row's leading 1 is in a column
            # without any other 1s; if so, add it to the matrix!
            leading_col = list(extra_row).index(1)
            if np.sum(rows[:, leading_col]) == 0:
                rref_mats_temp.append(np.vstack((rows, extra_row)))

    # Recursively run the function until enough rows have been added
    if current_num_rows == num_rows - 1:
        yield from rref_mats_temp
    elif current_num_rows < num_rows - 1:
        for new_rows in rref_mats_temp:
            yield from get_rref_matrices(num_rows, new_rows)


def check_independent(check_matrix):
    """
    Checks whether the generators, given by the check matrix
    check_matrix, are independent.

    We use the fact that the generators are independent iff the
    check vectors are linearly independent.

    Parameters
    ----------
    check_matrix : numpy.ndarray

    Returns
    -------
    independent : bool
        Whether or not the generators are independent.

    """

    n = check_matrix.shape[0]

    # For n = 1,2, the sets of generators are always independent
    # due to how we create them
    if n == 1 or n == 2:
        return True

    return binary_matrix_rank(check_matrix.copy()) == n  # TODO


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
        \Lambda = \\begin{pmatrix} 0 & I \\\ -I & 0 \end{pmatrix}.

    Parameters
    ----------
    check_matrix : numpy.ndarray

    Returns
    -------
    commute : bool
        Whether or not the generators commute.

    """

    n = check_matrix.shape[0]

    # Define the block matrix big_lambda
    I = np.eye(n, dtype=np.int8)
    O = np.zeros((n, n), dtype=np.int8)
    Lambda = np_block(((O, I), (-I, O)))

    # O = to_numpy_array(O_shared, n, n)
    # Lambda = to_numpy_array(Lambda_shared, 2*n, 2*n)

    m1 = check_matrix
    m2 = check_matrix.T
    tt = np.dot(m1, np.dot(Lambda, m2))

    return np.array_equal(
        # (check_matrix @ Lambda @ check_matrix.T) % 2, O
        tt % 2, O
    )


def check_valid(args):
    """
    Checks if the check matrix represents a valid set of generators
    for a maximal abelian subgroup of the n-qubit Pauli group.

    Parameters
    ----------
    args : tuple
        Contains the following elements:

        index : int, optional

        check_matrix : numpy.ndarray

    Returns
    -------
    None
        If the check matrix is not valid.
    check_matrix : numpy.ndarray
        check_matrix if it is valid.

    """

    if len(args) == 1:
        check_matrix = args[0]
    elif len(args) == 2:
        index, check_matrix = args

    # For testing
    if index % 10000 == 0:
        print(
            f'The check matrix that is being tested is\n'
            f'{check_matrix}'
        )

    with O_shared.get_lock(), Lambda_shared.get_lock():
        if check_commute(check_matrix):
            return check_matrix
        return None


# Some functions that allow numpy arrays to be shared between processes

def to_numpy_array(mp_arr, num_rows, num_cols):
    return np.frombuffer(mp_arr.get_obj()).reshape(num_rows, num_cols)


def init_shared_arrays(O_shared_, Lambda_shared_):
    global O_shared, Lambda_shared
    O_shared = O_shared_
    Lambda_shared = Lambda_shared_


# The main function
def get_max_abelian_subgroups(n):
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

    global O_shared, Lambda_shared

    start_time = time.perf_counter()

    subgroups = []

    # Define the block matrix big_lambda (and associated matrices)
    I = np.eye(n)

    O_shared = mp.Array(ctypes.c_int8, n**2)
    O = to_numpy_array(O_shared, n, n)
    O[:] = np.zeros((n, n))

    Lambda_shared = mp.Array(ctypes.c_int8, 4 * n**2)
    Lambda = to_numpy_array(Lambda_shared, 2*n, 2*n)
    Lambda[:] = np_block(((O, I), (-I, O)))

    # Consider every single n by 2n check matrix in reduced row echelon form,
    # and add to the list of subgroups if the generators
    # commute and are independent
    with mp.Pool(initializer=init_shared_arrays,
                 initargs=(O_shared, Lambda_shared)) as pool, \
            open('data/{n}_qubit_subgroups.data', 'ab') as writer:
        results = pool.imap(
            check_valid,
            enumerate(get_rref_matrices(n)),
            chunksize=1000
        )

        for item in results:
            if item is not None:
                subgroups.append(item)
                if len(subgroups) == 100_000:
                    pickle.dump(subgroups, writer)
                    subgroups = []

        pickle.dump(subgroups, writer)

    print(f'Total elapsed time: {time.perf_counter() - start_time}')

    return subgroups


if __name__ == '__main__':
    subgroups = get_max_abelian_subgroups(5)
