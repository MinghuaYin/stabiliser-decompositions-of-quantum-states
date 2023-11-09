#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import helper_functions as hf
import numpy as np

<<<<<<< HEAD
from subgroups import get_rref_matrices


def test_get_children():
=======
if __name__ == '__main__':
>>>>>>> 20931ef5297b6595435f71d3d65de5a5935159ca
    xmatr_aug = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]], dtype=np.int8)
    # support = hf.get_stab_support(xmatr_aug)
    child1, child2, rel_phase = hf.get_children(xmatr_aug)
<<<<<<< HEAD


def test_get_rref_matrices(n=6, repeats=0):
    for _ in range(repeats + 1):
        generator = get_rref_matrices(n)
        try:
            for _ in range(10_000):
                next(generator)
        except StopIteration:
            pass


def test_power(n):
    for _ in range(1000):
        2**n


def test_shift(n):
    for _ in range(1000):
        1 << n


if __name__ == '__main__':
    # start = time.perf_counter()
    # test_power(6)
    # print(f'Time elapsed: {time.perf_counter() - start}')

    # start = time.perf_counter()
    # # test_get_rref_matrices(6, 5)
    # test_shift(6)
    # print(f'Time elapsed: {time.perf_counter() - start}')

    start = time.perf_counter()
    test_get_rref_matrices(6, 5)
    print(f'Time elapsed: {time.perf_counter() - start}')
=======
>>>>>>> 20931ef5297b6595435f71d3d65de5a5935159ca
