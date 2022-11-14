#include <stdlib.h>

typedef struct neuron {
    int id;
    int num_synapses;
    struct neuron *sources;
    double *weights;
    double *delays;
    struct time *firing_times;
    double potential;
    double beta;
    double threshold;
    double refractory_period;
};

typedef struct network {
    int num_neurons;
    struct neuron *neurons;
};

void init_network(struct network **ptr, int num_neurons);
void init_state(struct network **ptr);
void clean_network(struct network **ptr);
double impulse_resp(double t, double beta);