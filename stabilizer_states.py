# Copied from old project

import math
import numba
import pickle
import sys
import time
import itertools as it
import multiprocessing as mp
import numpy as np
from numpy.linalg import norm
from scipy.linalg import null_space


np.set_printoptions(threshold=sys.maxsize)
eps = 1e-5
rnd_dec = 5


def normalize(state):
    return state / norm(state)


def n_fold_tensor_product(state, n):
    if n == 0:
        return np.eye(2**n)

    product = state
    for _ in range(n-1):
        product = np.kron(product, state)
    return product


def state_eq(state1, state2):
    diff = norm(state1 - state2)
    return diff < eps


def state_eq_up_to_phase(state1, state2):
    # print(f'Comparing {state1 = }, {state2 = }')
    if (state_eq(state1, np.zeros_like(state1))
            and not state_eq(state2, np.zeros_like(state2))) or \
       (state_eq(state2, np.zeros_like(state2))
            and not state_eq(state1, np.zeros_like(state1))):
        return False, np.nan
    elif state_eq(state1, state2):
        return True, 1
    else:
        state2_rndd = np.round(state2, decimals=rnd_dec)
        i = np.nonzero(state2_rndd)[0][0]
        state2_renorm = (state1[i] / state2[i]) * state2
        if state_eq(state1, state2_renorm):
            return True, state2[i] / state1[i]
        else:
            return False, np.nan


def get_matrix(check_vector, phase):
    """
    Converts a Pauli element's check vector into its matrix form.

    Parameters
    ----------
    check_vector : numpy.ndarray
        The check vector.
    phase : complex
        The phase of the element: should be :math:`\pm 1`.

    Returns
    -------
    matrix : numpy.ndarray
        The matrix representation of the Pauli element.
    operator_str : string
        The Pauli element in a readable format.

    """

    X = np.array([[0, 1],
                  [1, 0]])
    Y = np.array([[0, -1j],
                  [1j, 0]])
    Z = np.array([[1, 0],
                  [0, -1]])
    I = np.eye(2)

    pauli_matrices = {
        '00': I,
        '01': Z,
        '10': X,
        '11': Y
    }
    pauli_matrix_strs = {
        '00': 'I',
        '01': 'Z',
        '10': 'X',
        '11': 'Y'
    }

    n = int(len(check_vector) / 2)
    matrix = 1
    operator_str = '' if phase == 1 else '-'
    for i in range(n):
        # Figure out what the operator on the ith qubit is
        ith_operator = pauli_matrices[str(check_vector[i]) +
                                      str(check_vector[n+i])]
        matrix = np.kron(matrix, ith_operator)
        operator_str \
            += pauli_matrix_strs[str(check_vector[i]) +
                                 str(check_vector[n+i])]

    return (phase * matrix, operator_str)


def get_latex_for_state(state):
    """
    Generates the :math:`\LaTeX` for the given state.

    Parameters
    ----------
    state : numpy.ndarray
        The state in vector form.

    Returns
    -------
    latex : str

    """

    dim = len(state)
    n = int(np.log2(dim))
    support = np.count_nonzero(state)
    sign = {
        (1, False): '+',
        (-1, False): '-',
        (1, True): '',
        (-1, True): '-'
    }

    if support == 1:
        latex = ''
    elif math.sqrt(support) == int(math.sqrt(support)):
        latex = '\\frac{1}{%i}\\big{(}' % int(math.sqrt(support))
    else:
        latex = '\\frac{1}{\\sqrt{%i}}\\big{(}' % support

    is_first_term = True
    for k in range(dim):
        if state[k] == 0:
            continue

        if state[k].real != 0:
            latex += sign[(np.sign(state[k].real), is_first_term)]
        elif state[k].imag != 0:
            latex += '%si' % sign[(np.sign(state[k].imag), is_first_term)]
        bit_str = np.binary_repr(k, width=n)
        latex += '|%s\\rangle' % bit_str

        is_first_term = False
    if support != 1:
        latex += '\\big{)}'

    return latex


def state_from_subgroup(args):
    """
    Given an n-qubit check matrix and phase for each generator,
    this function finds the stabiliser state associated with
    this subgroup.

    Parameters
    ----------
    args : tuple
        Contains the following elements:

        index : int, optional

        n : int

        subgroup : numpy.ndarray
            The check matrix.

        phase_comb : tuple[int]

    Returns
    -------
    data : dict
        Stabiliser state data.

    """

    if len(args) == 3:
        n, subgroup, phase_comb = args
        index = None
    elif len(args) == 4:
        index, n, subgroup, phase_comb = args

    # Recover the generators in matrix form,
    # and also get readable strings of the generators
    generators = []
    generator_strs = []
    for i in range(n):
        data = get_matrix(
            subgroup[i, :],
            1 if phase_comb[i] == 0 else -1
        )
        generators.append(data[0])
        generator_strs.append(data[1])

    # Find all elements of the subgroup
    elements = []
    for powers in it.product([0, 1], repeat=n):
        element = np.eye(2**n)

        for i in range(n):
            if powers[i] == 1:
                element = element @ generators[i]

        elements.append(element)

    # Find the projection operator in matrix form
    P_S = np.zeros((2**n, 2**n), dtype=complex)
    for g in elements:
        P_S += g
    P_S *= 1 / 2**n

    # Find the 1-eigenvector of the projection operator,
    # i.e. the stabiliser state

    # stabiliser_state = np.around(null_space(P_S - np.eye(2**n)),
    #                              decimals=8)
    stabiliser_state = null_space(P_S - np.eye(2**n))

    # Renormalise so that the first amplitude is real and positive
    first_nonzero = np.flatnonzero(stabiliser_state.round(rnd_dec))[0]
    to_divide_by = stabiliser_state[first_nonzero, 0] / \
        abs(stabiliser_state[first_nonzero, 0])
    stabiliser_state = np.round(stabiliser_state / to_divide_by, decimals=8)

    # For testing
    if index is not None and index % 10000 == 0:
        print(f'Just found state with index {index}')

    try:
        return {
            'vector': stabiliser_state,
            # 'latex': get_latex_for_state(stabiliser_state),
            'check_matrix': subgroup,
            'generator_strs': generator_strs
            # 'vector': stabiliser_state.ravel()
        }
    except IndexError:
        print(f'Uh-oh, there\'s a problem with {stabiliser_state = }')
        print(f'{subgroup = }')
        print(f'{phase_comb = }')


def rref_binary(matrix):
    num_rows, num_cols = matrix.shape
    matrix = np.copy(matrix)

    row_to_comp = 0
    for j in range(num_cols):
        col = matrix[row_to_comp:, j]
        if np.count_nonzero(col) > 0:
            i = np.nonzero(col)[0][0] + row_to_comp
            matrix[(row_to_comp, i), :] = matrix[(i, row_to_comp), :]

            for ii in range(num_rows):
                if ii != row_to_comp and matrix[ii, j] != 0:
                    matrix[ii, :] ^= matrix[row_to_comp, :]

            row_to_comp += 1
            if row_to_comp == num_rows:
                break

    return matrix


def pauli_to_symplectic(pauli_string):
    x = []
    z = []
    for pauli in pauli_string:
        if pauli == 'I':
            x.append(0)
            z.append(0)
        elif pauli == 'X':
            x.append(1)
            z.append(0)
        elif pauli == 'Y':
            x.append(1)
            z.append(1)
        elif pauli == 'Z':
            x.append(0)
            z.append(1)
    return x + z


def paulistring_to_matrix(paulistring):
    X = np.array([[0, 1],
                  [1, 0]])
    Y = np.array([[0, -1j],
                  [1j, 0]])
    Z = np.array([[1, 0],
                  [0, -1]])
    I = np.eye(2)
    st_to_pauli_matrices = {
        'I': I,
        'Z': Z,
        'X': X,
        'Y': Y
    }
    Pm = []
    for x in paulistring:
        if len(Pm) == 0:
            Pm = st_to_pauli_matrices[x]
        else:
            Pm = np.kron(Pm, st_to_pauli_matrices[x])

    return Pm


def is_stab_from_paulis(state: np.ndarray, n):
    state = np.round(state, decimals=rnd_dec)

    # find all stabilizers of the state:

    pauli_labels = ['I', 'X', 'Y', 'Z']

    pauli_combinations = tuple(it.product(pauli_labels, repeat=n))

    stabset = []  # list of all stabilizers
    for labels in pauli_combinations:
        p_st = ''.join(labels)
        Pm = paulistring_to_matrix(p_st)

        resp = np.dot(Pm, state)
        resm = np.dot(-1 * Pm, state)

        if state_eq(resp, state):
            stabset.append(p_st)
        elif state_eq(resm, state):
            stabset.append('-'+p_st)

        if len(stabset) >= 64:
            return True

    # TODO Is there a way to return False sooner?
    return False


def stab_to_xmatr(state: np.ndarray, n):
    """
    Returns
    -------
    ind_matrix

    ind_signs
    """

    state = np.round(state, decimals=rnd_dec)

    # find all stabilizers of the state:

    pauli_labels = ['I', 'X', 'Y', 'Z']

    pauli_combinations = tuple(it.product(pauli_labels, repeat=n))

    stabset = []  # list of all stabilizers
    for labels in pauli_combinations:
        p_st = ''.join(labels)
        Pm = paulistring_to_matrix(p_st)

        resp = np.dot(Pm, state)
        resm = np.dot(-1 * Pm, state)

        if state_eq(resp, state):
            stabset.append(p_st)
        elif state_eq(resm, state):
            stabset.append('-'+p_st)

    # convert stabilizers into binary symplectic format
    signs = []  # 0 if +, 1 if -
    symp_matr = []
    for stab in stabset:
        symp_str = pauli_to_symplectic(stab)
        symp_int = [int(bit) for bit in symp_str]
        symp_matr.append(symp_int)
        if stab[0] == '-':
            signs.append(1)
        else:
            signs.append(0)
    # find linearly independent set (keep adding rows until the matrix has full rank, if a given row is lin dep with

    # keep adding rows which increase the rank of the check matrix
    # Q, R, P = scipy.linalg.qr(np.array(symp_matr).T, pivoting=True)
    # rank = np.linalg.matrix_rank(symp_matr)
    # ind_matrix = np.array(symp_matr)[P[:rank]]
    # ind_signs = np.array(signs)[P[:rank]]

    symp_matr = np.array(symp_matr)
    # print(symp_matr)

    # TODO Make this cleaner
    # Convert check matrix to rref, and find new phases
    ind_matrix = rref_binary(symp_matr)[:n, :]

    # if np.all(ind_matrix[-1, :] == 0):
    #     raise ValueError

    signs = []
    for row in ind_matrix:
        pauli, _ = get_matrix(row, 1)
        is_it, phase = state_eq_up_to_phase(state, pauli @ state)
        signs.append(0 if phase == 1 else 1)
    ind_signs = np.array(signs)
    return ind_matrix, ind_signs


def arr_to_hash(mat: np.ndarray, signs: np.ndarray):
    full_arr = np.copy(mat)
    full_arr = np.insert(full_arr, 0, signs, axis=1)
    stringhash = " / ".join(" ".join(map(str, row)) for row in full_arr)
    return stringhash


def stab_to_hash(state: np.ndarray, n):
    return arr_to_hash(*stab_to_xmatr(state, n))


def hash_to_arr(h: str, n):
    h = h.replace(' /', '')
    bin_list = h.split(' ')
    signs = np.array(bin_list[::2*n+1], dtype=int)
    xmatr_list = [int(bin_list[i]) for i in range(len(bin_list))
                  if i % (2*n+1) != 0]
    xmatr = np.array(xmatr_list).reshape(n, 2*n)
    return xmatr, signs


def get_stabiliser_states(n):
    """
    Finds all the n-qubit stabiliser states.

    Parameters
    ----------
    n : int

    Returns
    -------
    stab_state_sublist : list
        The (last 500,000) n-qubit stabiliser states.

    """

    start_time = time.perf_counter()

    # Generates string reps of all possible phases -
    # 0 means '+', 1 means '-'
    phase_combs = tuple(it.product([0, 1], repeat=n))

    try:
        with open(f'data/{n}_qubit_subgroups.data', 'rb') as reader:
            subgroups = pickle.load(reader)
    except FileNotFoundError:
        print(f'The file data/{n}_qubit_subgroups.data does not exist')
        return None

    # For each subgroup, find the 2^n stabiliser states
    # associated with it, accounting for the sign of each generator.
    # Add all found stabiliser states to a big list (may be split into chunks
    # if there are a lot of stabiliser states)
    args = ((index, n, subgroup, phase_comb)
            for index, (subgroup, phase_comb) in
            enumerate(it.product(subgroups, phase_combs)))

    with mp.Pool() as pool, \
            open(f'data/{n}_qubit_stab_states.data', 'wb') as writer:
        for _ in range(len(subgroups) * 2**n // 500_000 + 1):
            stab_state_sublist = pool.map(
                state_from_subgroup,
                it.islice(args, 500_000),
                chunksize=500
            )
            pickle.dump(stab_state_sublist, writer)

    print(f'Total elapsed time: {time.perf_counter() - start_time}')

    # Return the (last 500,000) n-qubit stabiliser states
    return stab_state_sublist


def get_stab_state_matrix(n):
    """
    Returns a :math:`\mathbb{C}^n \times |\mathcal{S_n}|`-sized matrix, where
    :math:`\mathcal{S_n}` is the set of all n-qubit stabiliser states,
    such that the columns are the stabiliser states with respect to the
    computational basis.

    Parameters
    ----------
    n : int

    Returns
    -------
    matrix : numpy.ndarray

    """

    start_time = time.perf_counter()

    stab_states = []
    try:
        with open(f'data/{n}_qubit_stab_states.data', 'rb') as reader:
            while True:
                stab_states += pickle.load(reader)
    except EOFError:
        pass

    matrix = np.hstack([state['vector'] for state in stab_states])

    with open(f'data/{n}_qubit_matrix.data', 'wb') as writer:
        pickle.dump(matrix, writer)
    # Save matrix in csv file
    np.savetxt(f'data/{n}_qubit_matrix.csv', matrix, delimiter=',', fmt='%.3e')

    print(f'Total elapsed time: {time.perf_counter() - start_time}')

    return matrix


if __name__ == '__main__':
    five_qubit_stabs = get_stabiliser_states(5)
