static PyObject *py_sim(PyObject *self, PyObject *args)
{
    PyObject *py_net;
    PyObject *py_weights;
    PyObject *py_delays;
    PyObject *py_sources;
    PyObject *py_initial_states;
    
    double t_max;
    double dt;
    double tol;

    if (!PyArg_ParseTuple(args, "OOOOOdd", &py_net, &py_weights, &py_delays, &py_sources, &py_initial_states, &t_max, &dt, &tol)) {
        return NULL;
    }

    struct network *net = (struct network *)PyCapsule_GetPointer(py_net, NULL);
    sim(net, py_weights, py_delays, py_sources, py_initial_states, t_max, dt, tol);

    Py_RETURN_NONE;
}