#include <stdlib.h>

#include "times.h"

double impulse_resp(double t, double beta) {
    if (t <= 0) return 0;
    return t / beta * exp(1 - t / beta);
}

void init_time(struct time **ptr)
{
    double s;
    struct time *tmp;
    *ptr = NULL;

    // Loop over all initial firing times of the neuron
    {
        tmp = malloc(sizeof(struct time));
        tmp->prev = *ptr;
        tmp->s = s;
        *ptr = tmp;
    }
}

void append_time(struct time **ptr, double s)
{
    struct time *tmp = malloc(sizeof(struct time));
    tmp->prev = *ptr;
    tmp->s = s;
    *ptr = tmp;
}

void pop_time(struct time **ptr)
{
    struct time *tmp = *ptr;
    *ptr = (*ptr)->prev;
    free(tmp);
}

void clean_time(struct time **ptr)
{
    while (*ptr != NULL) {
        pop(ptr);
    }
}