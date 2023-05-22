import numpy as np


def norm(x):
    return x / np.sum(x)

def mod(x, modulo, offset=0):
    return x - np.floor((x - offset) / modulo) * modulo

# def sphere_intersection(array, radius, length, time_step):
#     if array.size == 0:
#         return np.array([])
    
#     shift = array[0] - radius
#     shifted_array = array - shift
#     array = [np.arange(shifted_array[i] - radius, shifted_array[i] + radius + time_step, time_step) for i in range(shifted_array.size)]
#     return (np.concatenate(array) + shift) % length

# def sphere_intersection_complement(array, radius, length, time_step):
#     if array.size == 0:
#         return np.arange(0, length, time_step)

#     shift = array[0] - radius
#     shifted_array = array - shift
#     array = [np.arange(shifted_array[i] + radius + time_step, shifted_array[i+1] - radius, time_step) for i in range(shifted_array.size - 1)]
#     array += [np.arange(shifted_array[-1] + radius, length, time_step)]
#     return (np.concatenate(array) + shift) % length
