static PyMethodDef rsnnMethods[] = {
    {"sim", py_sim, METH_VARARGS, "Simulate the network in C."},
    {NULL, NULL, 0, NULL}
};