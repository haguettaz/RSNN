#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *
rsnn_sim(PyObject *self, PyObject *args)
{
    const struct 
}

// What we want is being able to do the following:
// 
// >>> import rsnn
// >>> % the three first can be implemented in Python
// >>> rsnn.simulation.network.init_network(num_neurons, num_synapses, weights, delays, and sources) % to initialize the network with weights, delays and sources.
// >>> rsnn.simulation.network.init_state() % to initialize the state of the network, i.e., the previous firing times of all neurons.
// >>> rsnn.simulation.network.clean_state() % to clean the state of the network, i.e., the previous firing times of all neurons.
// >>> rsnn.simulation.sim.sim() % to run a simulation of the network in C.
