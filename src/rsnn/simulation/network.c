#include <stdlib.h>

#include "network.h"
#include <math.h>


double impulse_resp(double t, double beta) {
    if (t <= 0) return 0;
    return t / beta * exp(1 - t / beta);
}

void init_network(struct network **ptr, int num_neurons){

}

void init_state(struct network **ptr){

}

void clean_network(struct network **ptr){

}