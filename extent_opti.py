#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numba
import itertools as it
import numpy as np

from F2_helper import mod2ham
from functools import reduce


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


@numba.jit(nopython=True, parallel=False)
def add_rows(a: np.ndarray, b: np.ndarray, n):
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
def rref_binary(xmatr_aug: np.ndarray):
    """
    'rref' function specifically for augmented check matrices.

    """

    num_rows, num_cols = xmatr_aug.shape
    xmatr_aug = np.copy(xmatr_aug)

    row_to_comp = 0
    for j in range(num_cols - 1):
        col = xmatr_aug[row_to_comp:, j]
        if np.count_nonzero(col) > 0:
            i = np.nonzero(col)[0][0] + row_to_comp
            temp = np.copy(xmatr_aug[i, :])
            xmatr_aug[i, :] = xmatr_aug[row_to_comp, :]
            xmatr_aug[row_to_comp, :] = temp

            for ii in range(num_rows):
                if ii != row_to_comp and xmatr_aug[ii, j] != 0:
                    # xmatr_aug[ii, :] ^= xmatr_aug[row_to_comp, :]
                    xmatr_aug[ii, :] = add_rows(xmatr_aug[ii, :],
                                                xmatr_aug[row_to_comp, :],
                                                num_rows)

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
    x_part = x_part[~np.all(x_part == 0, axis=1)].tolist()
    k = len(x_part)

    # If support is whole of F_2^n, then we're done
    if k == n:
        return np.array(range(2**n), dtype=np.int8)

    vector_space_basis = np.array([int(''.join(str(b) for b in bits), 2)
                                   for bits in x_part])
    vector_space = np.array([reduce(lambda x, y: x ^ y,
                                    (np.array(coeffs) * vector_space_basis).tolist())
                             for coeffs in it.product((0, 1), repeat=k)], dtype=np.int8)

    # Find a particular vector in the affine subspace
    pure_zs = xmatr_aug[k-n:, n:-1]
    pure_zs_numeric = np.array([int(''.join(str(b) for b in bits), 2)
                                for bits in pure_zs])
    signs = xmatr_aug[-pure_zs_numeric.shape[0]:, -1]

    # TODO Is this efficient?
    mod2ham_np = np.vectorize(mod2ham)
    for c in range(2**n):
        if np.array_equiv(mod2ham_np(c & pure_zs_numeric), signs):
            break

    return c ^ vector_space


def get_pauli_between_comp_states(start_bit: int, end_bit: int,
                                  x_part: np.ndarray) -> np.ndarray:
    """
    Given integers corresponding to comp basis state labels, and the 'X' part
    of a check matrix, find a Pauli that transforms the start comp state to
    the end comp state.

    """

    x_part = x_part[~np.all(x_part == 0, axis=1)]
    num_rows, n = x_part.shape
    start_vec = np.array(list(format(start_bit, f'0{n}b'))).astype(np.int8)
    end_vec = np.array(list(format(end_bit, f'0{n}b'))).astype(np.int8)

    for coeffs in it.product((0, 1), repeat=num_rows):
        coeffs = np.array(coeffs, dtype=np.int8).reshape(num_rows, 1)
        pauli = reduce(lambda r1, r2: r1 ^ r2, list(coeffs * x_part))
        if np.array_equiv(start_vec ^ pauli, end_vec):
            return pauli


def get_B_col(xmatr_aug: np.ndarray):
    """
    Split a state, given by an augmented check matrix, into two children.

    Parameters
    ----------
    xmatr_aug : np.ndarray
        The augmented check matrix (including signs) for the state.

    Returns
    -------


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
