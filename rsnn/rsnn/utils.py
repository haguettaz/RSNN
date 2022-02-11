import numpy as np


def is_memorized(net, data):
    for l in range(data.shape[1]):
        rolled_data = np.roll(data, -l, axis=-1)
        pred = net.forward(rolled_data[..., -net.Th :])
        true = rolled_data[..., 0]

        if np.any(np.not_equal(pred, true)):
            return False

    return True
