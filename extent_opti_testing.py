#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import extent_opti as eo
import numpy as np

if __name__ == '__main__':
    xmatr_aug = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]], dtype=np.int8)
    support = eo.get_stab_support(xmatr_aug)
