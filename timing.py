#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import timeit

import numpy as np


from subgroups import gf2elim
from stabilizer_states import rref_binary

mat = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]], dtype=np.int8)


if __name__ == '__main__':
    print(timeit.timeit('gf2elim(np.copy(mat))',
          globals=globals(), number=100_000))
    print(timeit.timeit('rref_binary(np.copy(mat))',
          globals=globals(), number=100_000))
    print('----------')
