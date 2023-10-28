#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import timeit

import numpy as np


def fake_log(n):
    match n:
        case 64:
            k = 6
        case 32:
            k = 5
        case 16:
            k = 4
        case 8:
            k = 3
        case 4:
            k = 2
        case 2:
            k = 1
        case 1:
            return True
        case _:
            return False

    return k


answers = dict((2**k, k) for k in range(12))


def fake_log_2(n):
    try:
        return answers[n]
    except KeyError:
        return False


if __name__ == '__main__':
    print(timeit.timeit('fake_log(64)', globals=globals(), number=10000))
    print(timeit.timeit('math.log(64, 2)', globals=globals(), number=10000))
    print(timeit.timeit('answers[64]', globals=globals(), number=10000))
    print(timeit.timeit('fake_log_2(64)', globals=globals(), number=10000))
    print('----------')
