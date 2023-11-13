#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import helper_functions as hf
import numpy as np

from subgroups import get_rref_matrices, check_commute


def test_get_children():
    xmatr_aug = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]], dtype=np.int8)
    # support = hf.get_stab_support(xmatr_aug)
    child1, child2, rel_phase = hf.get_children(xmatr_aug)


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


def test_rref_binary():
    xmatr_aug = np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                          [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                          [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.int8)
    hf.rref_binary(xmatr_aug)


def test_check_commute():
    xmatr = np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                      [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [1, 0, 1, 1, 0, 0, 0, 1, 0, 0]], dtype=np.int8)
    check_commute(xmatr)


def test(function, args):
    start = time.perf_counter()
    for _ in range(10):
        function(*args)
    print(f'Time elapsed: {time.perf_counter() - start}')


if __name__ == '__main__':
    test(test_check_commute, [])
