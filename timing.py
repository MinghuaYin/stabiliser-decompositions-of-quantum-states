#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import timeit

import numpy as np

mat = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]], dtype=np.int8)


if __name__ == '__main__':
    # print(timeit.timeit('eo.rref_binary(np.random.rand(6, 12).round().astype(np.int8))',
    #       globals=globals(), number=100_000))
    # print(timeit.timeit('gf2elim(np.random.rand(6, 12).round().astype(np.int8))',
    #       globals=globals(), number=100_000))
    # print(timeit.timeit('stabs.rref_binary(np.random.rand(6, 12).round().astype(np.int8))',
    #       globals=globals(), number=100_000))
    print('----------')
