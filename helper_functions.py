#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numba

import F2_helper.F2_helper as f2
import multiprocessing as mp
import numpy as np
import scipy.sparse as spr

from functools import reduce
from itertools import product
from typing import List, Tuple

n = 6

sqrt2 = 1.4142135623730950

# -------- Functions for generating the B matrix of linearly dependent triples --------


@numba.jit(nopython=True, parallel=False)
def np_unique_impl(a):
    # https://github.com/numba/numba/issues/2884#issuecomment-382278786
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
def rref_binary(mat: np.ndarray):
    num_rows, num_cols = mat.shape
    mat = np.copy(mat)

    row_to_comp = 0
    for j in range(2*num_rows):
        col = mat[row_to_comp:, j]
        if np.count_nonzero(col) > 0:
            i = np.nonzero(col)[0][0] + row_to_comp
            temp = np.copy(mat[i, :])
            mat[i, :] = mat[row_to_comp, :]
            mat[row_to_comp, :] = temp

            for ii in range(num_rows):
                if ii != row_to_comp and mat[ii, j] != 0:
                    mat[ii, :] ^= mat[row_to_comp, :]

            row_to_comp += 1
            if row_to_comp == num_rows:
                break

    return mat


@numba.jit(nopython=True, parallel=False)
def rref_binary_aug(xmatr_aug: np.ndarray):
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

    i = np.argmax(xmatr_aug[0, :])

    new_row = np.zeros(2*n + 1, dtype=np.int8)
    new_row[i + n] = 1

    # Find children
    child1 = np.vstack((new_row, xmatr_aug[1:, :]))
    new_row[-1] = 1
    child2 = np.vstack((new_row, xmatr_aug[1:, :]))

    child1 = rref_binary_aug(child1)
    child2 = rref_binary_aug(child2)

    # TODO A 'more clever' way to put child matrices in 'rref' ---
    # however, for this to work, check_matrices.py needs to be modified
    # (bottom right block in a different form)
    # nz_rows = np.nonzero(child1[:, i+n])[0].tolist()
    # for r in nz_rows[1:]:
    #     child1[r, :] = add_rows(child1[r, :], child1[0, :], n)
    #     child2[r, :] = add_rows(child2[r, :], child2[0, :], n)

    # pivots = list(zip(range(n), np.argmax(child1, axis=1).tolist()))
    # pivots.sort(key=lambda x: x[1])
    # reordering = tuple(x[0] for x in pivots)
    # child1[:, :] = child1[reordering, :]
    # child2[:, :] = child2[reordering, :]

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


def find_B_data_for_one_xmatr(args) -> List:
    """

    Parameters
    ----------
    xmatr : np.ndarray

    n : int

    index : int

    Results
    -------
    results : List
        Consists of tuples of the form

        col_num : int
        child1 : np.ndarray
        child2 : np.ndarray
        rel_phase : complex

    """

    xmatr, n, index = args

    results = []

    for numeric in range(1 << n):
        col_num = index * (1 << n) + numeric
        if col_num % 10_000_000 == 0:
            print(col_num)
            # print(time.perf_counter())

        signs = f2.int_to_array(numeric, n)
        child1, child2, rel_phase = get_children(
            np.column_stack((xmatr, signs)))
        results.append((col_num, child1, child2, rel_phase))

    return results
