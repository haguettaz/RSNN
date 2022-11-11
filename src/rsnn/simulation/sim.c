// need to store the firing times of all neurons
// class neuron and class network
// class neuron: weigts, delays, sources, firing times
// class network: neurons,


#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "times.h"
#include "network.h"
#include "sim.h"


void init(struct time **ptr) {
    
}

void sim_step(struct network *net, double t_max, double t, double tol)
{
    struct time *ptr = NULL;
    double tmp = 0;
    double contrib = 0;

    for (int l=0; l<net->num_neurons; l++) 
        {
            if (t - net->neurons[l].firing_times->s < net->neurons[l].refractory_period) continue;
            for (int k=0; k<net->neurons[l].num_synapses; k++) 
            {
                if (fabs(net->neurons[l].weights[k]) < tol) continue; // skip if weight is close to 0, i.e., there is virtually no connection
                
                ptr = net->neurons[l].sources[k].firing_times;
                contrib = 0;
                while (ptr != NULL) {
                    tmp = t - ptr->s - net->neurons[l].delays[k];
                    if (tmp <= 0) {
                        ptr = ptr->prev; // impulse response being causal, the state of the neuron does not depend on future spikes
                        continue;
                    }
                    tmp = impulse_resp(tmp, net->neurons[l].beta);
                    contrib += tmp;
                    if (tmp < tol) break; // firing times are inserted in ascending order, so if the contribution of the current spike is smaller than the tolerance, the contributions of all previous spikes will be smaller as well
                    ptr = ptr->prev;
                }
                net->neurons[l].potential += net->neurons[l].weights[k] * contrib;
            }

            if (net->neurons[l].potential > net->neurons[l].threshold) {
                append_time(&(net->neurons[l].firing_times), t);
            }
        }
}

void sim(struct network *net, weights, delays, sources, initial_states, double t_max, double dt, double tol)
{
    // weights, delays, sources, and initial_states are all list of list, the first dimension indexing the neuron 
    double t = 0;

    // initialize the network with weights, delays, sources, and initial_states of all neurons
    init_network(net, weights, delays, sources);
    init_state(net, initial_states);

    while (t < t_max) {
        sim_step(net, t_max, t, tol);
        t += dt;
    }

    // need to return the firing times of all neurons
}