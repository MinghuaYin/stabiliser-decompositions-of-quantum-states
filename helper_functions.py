#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numba
import pickle
import time

import F2_helper as f2
import numpy as np
import scipy.sparse as spr

from functools import reduce
from itertools import product
from typing import List, Tuple


# https://github.com/numba/numba/issues/2884#issuecomment-382278786
@numba.jit(nopython=True, parallel=False)
def np_unique_impl(a):
    b = np.sort(a.flatten())
    unique = list(b[:1])
    counts = [1 for _ in unique]
    for x in b[1:]:
        if x != unique[-1]:
            unique.append(x)
            counts.append(1)
        else:
            counts[-1] += 1
    return np.array(unique), np.array(counts)


# TODO Rephrase things in terms of dot products? (Would this even be any better?)
@numba.jit(nopython=True, parallel=False)
def add_rows(a: np.ndarray, b: np.ndarray, n: int):
    """
    Add one row of an augmented check matrix (b) to another row (a),
    dealing with the sign correctly.

    """

    # 0: I, 1: X, 3: Z, 4: Y
    a_mod_4 = a[:n] + 3*a[n:-1]
    b_mod_4 = b[:n] + 3*b[n:-1]

    where_id = (a_mod_4 * b_mod_4 == 0)
    double_paulis = a_mod_4 - b_mod_4
    double_paulis[where_id] = 0

    # numba doesn't support np.unique 😭
    vals, counts = np_unique_impl(double_paulis)

    # phase_correction = np.sum(counts[vals != 0]) // 2
    phase_correction = np.sum(np.where(vals != 0, counts, 0)) // 2
    # phase_correction += np.sum(
    #     counts[(vals == 2) | (vals == 1) | (vals == -3)])
    phase_correction += np.sum(
        np.where((vals == 2) | (vals == 1) | (vals == -3), counts, 0))

    result = a ^ b
    result[-1] ^= phase_correction % 2
    return result


@numba.jit(nopython=True, parallel=False)
def rref_binary(xmatr_aug: np.ndarray, augmented=True):
    """
    'rref' function specifically for augmented check matrices.

    """

    num_rows, num_cols = xmatr_aug.shape
    xmatr_aug = np.copy(xmatr_aug)

    row_to_comp = 0
    for j in range(2*num_rows):
        col = xmatr_aug[row_to_comp:, j]
        if np.count_nonzero(col) > 0:
            i = np.nonzero(col)[0][0] + row_to_comp
            temp = np.copy(xmatr_aug[i, :])
            xmatr_aug[i, :] = xmatr_aug[row_to_comp, :]
            xmatr_aug[row_to_comp, :] = temp

            for ii in range(num_rows):
                if ii != row_to_comp and xmatr_aug[ii, j] != 0:
                    if augmented:
                        xmatr_aug[ii, :] = add_rows(xmatr_aug[ii, :],
                                                    xmatr_aug[row_to_comp, :],
                                                    num_rows)
                    else:
                        xmatr_aug[ii, :] ^= xmatr_aug[row_to_comp, :]

            row_to_comp += 1
            if row_to_comp == num_rows:
                break

    return xmatr_aug


def get_stab_support(xmatr_aug: np.ndarray) -> np.ndarray:
    """
    Given an augmented check matrix (which uniquely determines a stab state),
    find the support of the stab state.

    """

    n = xmatr_aug.shape[0]
    x_part = xmatr_aug[:, :n]
    x_part = x_part[~np.all(x_part == 0, axis=1)]
    k = x_part.shape[0]

    # If support is whole of F_2^n, then we're done
    if k == n:
        return np.array(range(1 << n))

    vector_space_basis = np.array([f2.array_to_int(bits) for bits in x_part])
    if vector_space_basis.size == 0:
        vector_space = 0
    else:
        vector_space = np.array([reduce(lambda x, y: x ^ y,
                                        (np.array(coeffs) * vector_space_basis).tolist())
                                 for coeffs in product((0, 1), repeat=k)])

    # Find a particular vector in the affine subspace
    pure_zs = xmatr_aug[k-n:, n:-1]
    pure_zs_numeric = np.array([f2.array_to_int(bits) for bits in pure_zs])
    signs = xmatr_aug[-pure_zs_numeric.shape[0]:, -1]

    # TODO Is this efficient?
    mod2ham_np = np.vectorize(f2.mod2ham)
    for c in range(1 << n):
        if np.array_equiv(mod2ham_np(c & pure_zs_numeric), signs):
            break

    return c ^ vector_space


def get_pauli_between_comp_states(start_bit: int, end_bit: int,
                                  xmatr_aug: np.ndarray) -> np.ndarray:
    """
    Given integers corresponding to comp basis state labels, 
    and a check matrix, find a Pauli that transforms the start comp state to
    the end comp state.

    """

    n = xmatr_aug.shape[0]
    x_part = xmatr_aug[:, :n]
    xmatr_aug = xmatr_aug[~np.all(x_part == 0, axis=1)]
    num_rows = xmatr_aug.shape[0]

    start_vec = f2.int_to_array(start_bit, n)
    end_vec = f2.int_to_array(end_bit, n)

    for coeffs in product((0, 1), repeat=num_rows):
        coeffs = np.array(coeffs, dtype=np.int8).reshape(num_rows, 1)
        pauli = reduce(lambda r1, r2: add_rows(r1, r2, n),
                       list(coeffs * xmatr_aug))
        if np.array_equiv(start_vec ^ pauli[:n], end_vec):
            return pauli


def get_children(xmatr_aug: np.ndarray) -> Tuple[np.ndarray, np.ndarray, complex]:
    """
    Split a stab state, given by an augmented check matrix, into two children.

    Parameters
    ----------
    xmatr_aug : np.ndarray
        The augmented check matrix (including signs) for the stab state.

    Returns
    -------
    child1 : np.ndarray

    child2 : np.ndarray

    rel_phase : complex

    """

    n = xmatr_aug.shape[0]

    i = np.argmax(xmatr_aug[0, :])

    new_row = np.zeros(2*n + 1, dtype=np.int8)
    new_row[i + n] = 1

    # Find children
    child1 = np.vstack((new_row, xmatr_aug[1:, :]))
    new_row[-1] = 1
    child2 = np.vstack((new_row, xmatr_aug[1:, :]))

    child1 = rref_binary(child1)
    child2 = rref_binary(child2)

    # Find the 'lowest' label in each child's support. Our convention is that
    # each stab state is normalized such that the amplitude of the component
    # with the lowest label is real and positive
    support1 = get_stab_support(child1)
    support2 = get_stab_support(child2)
    lowest_lab_1 = np.min(support1)
    lowest_lab_2 = np.min(support2)

    # Swap children if child2 has a lower lowest label than child1
    if lowest_lab_2 < lowest_lab_1:
        child1, child2 = child2, child1
        support1, support2 = support2, support1
        lowest_lab_1, lowest_lab_2 = lowest_lab_2, lowest_lab_1

    # Find phase
    pauli = get_pauli_between_comp_states(lowest_lab_1, lowest_lab_2,
                                          xmatr_aug)
    # Get phase in front of Pauli
    rel_phase = 1 - 2*pauli[-1]

    # Get '-1 contributions' from Z part
    z_part = pauli[n:-1]
    rel_phase *= f2.sign_mod2product(lowest_lab_1, f2.array_to_int(z_part))

    # Get 'i' contributions from Y's
    x_part = pauli[:n]
    powers_of_i = (1, 1j, -1, -1j)
    y_is_there = np.where(x_part + z_part == 2, 1, 0)
    rel_phase *= powers_of_i[np.sum(y_is_there) % 4]

    return child1, child2, rel_phase


def get_B(xmatr_aug_list: List[np.ndarray], n: int) -> spr.csc_array:
    """
    Get a basis of triples in matrix form given a list of
    augmented check matrices that have been ordered by increasing support size.

    """

    B = spr.dok_array((len(xmatr_aug_list), len(xmatr_aug_list) - (1 << n)),
                      dtype=complex)

    hash_map = dict((str(mat), i) for i, mat in
                    enumerate(xmatr_aug_list[: 1 << n]))

    for col_num, xmatr_aug in enumerate(xmatr_aug_list[1 << n:]):
        if col_num % 100_000 == 0:
            print(col_num)

        # Hash all the augmented check matrices as we go through them
        hash_map[str(xmatr_aug)] = col_num + (1 << n)

        child1, child2, rel_phase = get_children(xmatr_aug)
        child1_index = hash_map[str(child1)]
        child2_index = hash_map[str(child2)]

        B[child1_index, col_num] = 1
        B[child2_index, col_num] = rel_phase
        B[col_num + (1 << n), col_num] = -math.sqrt(2)

    return B.tocsc()


def main(n):
    with open(f'data/{n}_qubit_subgroups.data', 'rb') as reader:
        xmatr_list = pickle.load(reader)

    print(f'{len(xmatr_list) = }')

    # Generate augmented matrices
    xmatr_aug_list = []
    for mat in xmatr_list:
        for numeric in range(1 << n):
            signs = f2.int_to_array(numeric, n)
            xmatr_aug_list.append(np.column_stack((mat, signs)))

    return get_B(xmatr_aug_list, n)


if __name__ == '__main__':
    start = time.perf_counter()
    n = 5
    B = main(n)
    spr.save_npz(f'data/{n}_qubit_B', B)
    print(f'Time elapsed: {time.perf_counter() - start}')
