from math import *

import numpy as np


def divisors(n):

    i = 1
    lst = []
    while i * i < n:
        if n % i == 0:
            lst.append(i)
        i += 1

    for i in range(int(sqrt(n)), 0, -1):
        if n % i == 0:
            lst.append(n // i)

    return lst


def is_prime(n):
    return len(divisors(n)) == 2


def cyclic_range(idx_start, length, idx_max):
    if isinstance(idx_start, int):
        idx_start = np.array([idx_start])

    indices = idx_start[:, None] + np.arange(length)[None, :]
    return indices % idx_max
