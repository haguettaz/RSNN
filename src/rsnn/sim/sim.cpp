// #include <torch/extension.h>

// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <random>

// using namespace std;

// double DELTA = 1e-5;

// double impulse_resp(double t, double beta) {
//   if (t <= 0) return 0;
//   return (t / beta) * exp(1 - t / beta);
// }

// vector<vector<double>> sim_cpp(double t0, double tmax, vector<vector<double>> spikes, torch::Tensor sources, torch::Tensor delays, torch::Tensor weights, double Tr, double beta, double theta, double etab)
// {
//   int L = sources.size(0);
//   int K = sources.size(1);

//   double t = t0; // simulation time

//   // Adaptive time step
//   // Note: the potential variation in-between two consecutive time steps is upper bounded by K * wb * exp(1) / beta
//   // In probability, we also have to include the spiking rate of the network, which can be approximated by 1 / Tr
//   // We get the following upper bound for the time step: K * wb * exp(1) / (beta * Tr)
//   // double counts_step = beta * Tr / (exp(1) * DT * K * wb);
  
//   vector<int> counts(L, 0);
//   vector<double> counts_step(L, etab);
//   vector<int>::iterator ptr_counts;
//   vector<double>::iterator ptr_counts_step;

//   double z; // neuron potential
//   double tmp; // temporary time variable

//   vector<vector<double>>::iterator ptr_spikes; // iterator for firing times
//   vector<double>::reverse_iterator ptr_spikes_sources; // iterator for sources firing times
//   int* ptr_sources; // pointer to sources
//   double* ptr_delays; // pointer to delays
//   double* ptr_weights; // pointer to weights

//   std::default_random_engine generator;
//   normal_distribution<double> distribution{0, etab/10};

//   // init counts_step for neuron dependent adaptive time step
//   ptr_weights = weights.data_ptr<double>();
//   ptr_counts_step = counts_step.begin();
//   for (int l = 0; l < L; l++) {
//     for (int k=0; k<K; k++) {
//       if (*ptr_weights > 0) *ptr_counts_step += *ptr_weights;
//       ptr_weights++;
//     }
//     *ptr_counts_step *= DELTA * exp(1) / beta;
//     ptr_counts_step++;
//   }

//   do
//   {
//     ptr_counts = counts.begin();
//     ptr_counts_step = counts_step.begin();
//     ptr_spikes = spikes.begin(); // init iterator for neuron firing times
//     ptr_weights = weights.data_ptr<double>(); // init pointer to weights
//     ptr_delays = delays.data_ptr<double>(); // init pointer to delays
//     ptr_sources = sources.data_ptr<int>(); // init pointer to sources

//     // For each neuron
//     for(int l=0; l<L; l++)
//     {
//       // if the neuron is still driven by an external input
//       if (ptr_spikes->size() && (*(ptr_spikes->end() - 1) >= t)) 
//       {
//         ptr_spikes++, ptr_counts++, ptr_counts_step++;
//         ptr_weights += K, ptr_delays += K, ptr_sources += K;
//         continue;
//       }

//       // if the neuron cannot be about to spike
//       if (*ptr_counts > 0)
//       {
//         (*ptr_counts)--;
//         ptr_spikes++, ptr_counts++, ptr_counts_step++;
//         ptr_weights += K, ptr_delays += K, ptr_sources += K;
//         continue;
//       }

//       // if the neuron is still recovering from its last spike
//       if (ptr_spikes->size() && (t - *(ptr_spikes->end() - 1) <= Tr)) 
//       {
//         ptr_spikes++, ptr_counts++, ptr_counts_step++;
//         ptr_weights += K, ptr_delays += K, ptr_sources += K;
//         continue;
//       }

//       // the noise on the potential is a white (truncated) Gaussian noise
//       do
//       {
//         z = distribution(generator);
//       } while (z > etab);

//       for (int k=0; k<K; k++)
//       {
//         ptr_spikes_sources = spikes[*ptr_sources].rbegin();
//         while (ptr_spikes_sources != spikes[*ptr_sources].rend())
//         {
//           tmp = t - *ptr_spikes_sources - *ptr_delays;
//           if (tmp > Tr) break;
//           z += (*ptr_weights) * impulse_resp(tmp, beta);
//           ptr_spikes_sources++;
//         }
//         ptr_weights++, ptr_delays++, ptr_sources++;
//       }

//       if (z >= theta) ptr_spikes->push_back(t); // neuron is spiking
//       else *ptr_counts = (int) ((theta - z) / *ptr_counts_step); // adaptive time step

//       ptr_spikes++, ptr_counts++, ptr_counts_step++;
//     }
//     t+=DELTA;
//   } while (t < tmax);

//   cout << "Simulation completed." << endl;

//   return spikes;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("sim_cpp", &sim_cpp, "Simulation");
// }