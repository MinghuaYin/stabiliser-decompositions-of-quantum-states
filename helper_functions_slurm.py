#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import pickle

import F2_helper.F2_helper as f2
import multiprocessing as mp
import scipy.sparse as spr

from typing import List
from io import BufferedReader

from helper_functions import *

# tail = ''
tail = '_real'


def get_B_data():
    """
    Get a basis of triples in matrix form given a list of (not augmented)
    check matrices that have been ordered by increasing support size.

    """

    with open(f'data/{n}_qubit_subgroups{tail}.data', 'rb') as r1:
        xmatr_list = pickle.load(r1)

    list_length = len(xmatr_list)
    print(f'{list_length = }')

    with mp.Pool() as pool, \
            open(f'data/{n}_B_data{tail}.data', 'ab') as writer:
        lists_of_results = pool.imap_unordered(
            find_B_data_for_one_xmatr,
            ((xmatr, n, index)
             for index, xmatr in enumerate(xmatr_list[1:])),
            chunksize=100)

        B_data = []
        for results in lists_of_results:
            B_data += results
            if len(B_data) >= 100_000:
                pickle.dump(B_data, writer)
                B_data = []

        pickle.dump(B_data, writer)

    print('get_B_data finished')


def update_data(args):
    data, hash_map = args
    updated_data = []

    for col_num, child1, child2, rel_phase in data:
        child1_index = hash_map[str(child1[:, :-1])] * (1 << n) \
            + f2.array_to_int(child1[:, -1])
        child2_index = hash_map[str(child2[:, :-1])] * (1 << n) \
            + f2.array_to_int(child2[:, -1])
        updated_data.append((col_num, child1_index, child2_index, rel_phase))

    return updated_data


def args(reader: BufferedReader, hash_map: dict):
    try:
        while True:
            yield (pickle.load(reader), hash_map)
    except EOFError:
        pass


def get_dict_form_data():
    """
    Get a dictionary of data that is ready to be inserted into a sparse matrix.

    """

    with open(f'data/{n}_qubit_hash_map{tail}.data', 'rb') as r1, \
            open(f'data/{n}_B_data{tail}.data', 'rb') as r2, \
            mp.Pool() as pool:
        hash_map = pickle.load(r1)
        print(f'{len(hash_map) = }')

        big_dict = {}
        results = pool.imap_unordered(
            update_data, args(r2, hash_map), chunksize=20)
        for partial_data in results:
            for col_num, child1_index, child2_index, rel_phase in partial_data:
                big_dict[(child1_index, col_num)] = 1
                big_dict[(child2_index, col_num)] = rel_phase
                big_dict[(col_num + (1 << n), col_num)] = -sqrt2

    with open(f'data/{n}_qubit_dict_form{tail}.data', 'wb') as w:
        pickle.dump(big_dict, w)


def from_dict_form_data(num_of_stab_states):
    dtype = float if tail == '_real' else complex
    B = spr.dok_array((num_of_stab_states, num_of_stab_states - (1 << n)),
                      dtype=dtype)

    with open(f'data/{n}_qubit_dict_form{tail}.data', 'rb') as r:
        big_dict = pickle.load(r)

    for k, v in big_dict.items():
        B[k] = v

    B = B.tocsc()
    spr.save_npz(f'data/{n}_qubit_B{tail}_stabs', B)
    return B


if __name__ == '__main__':
    # get_B_data()
    get_dict_form_data()
    # from_dict_form_data(315057600)
