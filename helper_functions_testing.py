#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import helper_functions as hf
import numpy as np

if __name__ == '__main__':
    xmatr_aug = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]], dtype=np.int8)
    # support = hf.get_stab_support(xmatr_aug)
    child1, child2, rel_phase = hf.get_children(xmatr_aug)
