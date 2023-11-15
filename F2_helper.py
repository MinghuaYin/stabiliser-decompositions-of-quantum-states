import functools
import numpy as np
from typing import List


def evaluate_poly(non_zero_coeffs: List[int], integer: int) -> int:
    terms = [integer & index == index for index in non_zero_coeffs]
    return int(functools.reduce(lambda x, y: x ^ y, terms, 0))


def sign_evaluate_poly(non_zero_coeffs: List[int], integer: int) -> int:
    return 1-2*evaluate_poly(non_zero_coeffs, integer)


def mod2ham(x: int) -> int:
    parity = 0

    while x:
        parity ^= 1
        x &= x - 1

    return parity


def mod2product(x: int, y: int) -> int:
    return mod2ham(x & y)


def sign_mod2product(x: int, y: int) -> int:
    return 1-2*mod2product(x, y)


def imag_mod2product(x: int, y: int) -> complex:
    return 1 + (1j-1)*mod2product(x, y)


def num_bin_digits(x: int) -> int:
    return len(bin(x)[2:])


def fast_log2(x: int) -> int:
    return num_bin_digits(x) - 1


def get_bit_at(x: int, bit_location: int) -> int:
    power_of_two = 1 << bit_location

    return (x & power_of_two) == power_of_two


def get_vector_expansion(dimension: int, basis_vectors: List[int], coefficients: int) -> int:
    terms = [get_bit_at(coefficients, j)*basis_vectors[j]
             for j in range(dimension)]
    return functools.reduce(lambda x, y: x ^ y, terms, 0)


def array_to_int(array: np.ndarray) -> int:
    return int(''.join(str(b) for b in array.tolist()), 2)


def int_to_array(numeric: int, n: int) -> np.ndarray:
    return np.array(list(format(numeric, f'0{n}b'))).astype(np.int8)
