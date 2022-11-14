import torch

import sim_cpp

def sim(sources, delays, weights, inputs):
    output = sim_cpp.sim(sources, delays, weights, inputs)
    return output # or save output to a file?