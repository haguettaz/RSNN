static struct PyModuleDef rsnnmodule = {
    PyModuleDef_HEAD_INIT,
    "rsnn",   /* name of module */
    "C library for fast network simulation",     /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    rsnnMethods
};