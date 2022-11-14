import time

from torch.utils.cpp_extension import load

sim_cpp = load(name="sim_cpp", sources=["src/rsnn/sim/sim.cpp"], verbose=True)

def sim(sources, delays, weights, max_t, dt):
    start = time.time()
    sim_cpp.sim_cpp(sources, delays, weights, max_t, dt)
    end = time.time()
    print("sim_cpp.sim took", end - start, "seconds")

    start = time.time()
    t = 0
    while t < max_t:
        t += dt
    end = time.time()
    print("while loop took", end - start, "seconds")