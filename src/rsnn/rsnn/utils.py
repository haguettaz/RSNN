import pickle


def circle_pairwise_generator(iterable):
    length = len(iterable)
    for i in range(length):
        yield iterable[i], iterable[(i + 1) % length]


def firing_times_generator(firing_times, duration):
    for t in firing_times:
        yield t % duration


def active_times_generator(firing_times, active_min, active_max, step, duration):
    for ft in firing_times:
        t = ft
        while t >= ft + active_min:
            yield t % duration
            t -= step

        t = ft + step
        while t <= ft + active_max:
            yield t % duration
            t += step


def silent_times_generator(firing_times, active_min, refractory_period, step, duration):
    for ft1, ft2 in circle_pairwise_generator(firing_times):
        while ft2 < ft1:
            ft2 += duration

        t = ft1 + refractory_period
        while t <= ft2 + active_min:
            yield t % duration
            t += step

def save(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)