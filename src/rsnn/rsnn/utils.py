import os
import pickle

# import numpy as np

# def circle_pairwise_generator(iterable):
#     length = len(iterable)
#     for i in range(length):
#         yield iterable[i], iterable[(i + 1) % length]


# def times_generator(times, modulo):
#     for t in times:
#         yield t % modulo


# def surrounding_times_generator(times, left, right, step, modulo):
#     for t1 in times:
#         for t in np.arange(t1 + left, t1 + right + step, step):
#             yield t % modulo


# def complement_surrounding_times_generator(times, left, right, step, modulo):
#     for t1, t2 in circle_pairwise_generator(times):
#         while t2 < t1:
#             t2 += modulo

#         for t in np.arange(t1 + right + step, t2 + left, step):
#             yield t % modulo

def save(obj, filename):
    if os.path.dirname(filename) != '':
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)