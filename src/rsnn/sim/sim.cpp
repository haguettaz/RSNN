#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

double DT = 1e-5;

double impulse_resp(double t, double beta) {
  if (t <= 0) return 0;
  return (t / beta) * exp(1 - t / beta);
}

vector<vector<double>> sim_cpp(double max_t, vector<vector<double>> firing_times, torch::Tensor sources, torch::Tensor delays, torch::Tensor weights, double Tr, double beta, double theta, double eta, uint seed)
{

  int L = sources.size(0);
  int K = sources.size(1);

  double t = 0; // simulation time

  // Adaptive time step
  // Note: the potential variation in-between two consecutive time steps is upper bounded by K * wb * exp(1) / beta
  // In probability, we also have to include the spiking rate of the network, which can be approximated by 1 / Tr
  // We get the following upper bound for the time step: K * wb * exp(1) / (beta * Tr)
  // double gamma_c = beta * Tr / (exp(1) * DT * K * wb);

  
  vector<int> c(L, 0);
  vector<double> gamma_c(L, 0);
  vector<int>::iterator ptr_c;
  vector<double>::iterator ptr_gc;

  double z; // neuron potential
  double tmp; // temporary time variable

  vector<vector<double>>::iterator ptr_f; // iterator for firing times
  vector<double>::reverse_iterator ptr_fs; // iterator for sources firing times
  double* ptr_w; // pointer to weights
  double* ptr_d; // pointer to delays
  int* ptr_s; // pointer to sources

  mt19937 gen{seed};
  normal_distribution<double> d{0, eta};

  // init gamma_c for neuron dependent adaptive time step
  ptr_w = weights.data_ptr<double>();
  ptr_gc = gamma_c.begin();
  for (int l = 0; l < L; l++) {
    for (int k=0; k<K; k++) {
      if (*ptr_w > 0) *ptr_gc += *ptr_w;
      ptr_w++;
    }
    *ptr_gc *= DT * exp(1) / beta;
    ptr_gc++;
  }

  do
  {
    ptr_c = c.begin();
    ptr_gc = gamma_c.begin();
    ptr_f = firing_times.begin(); // init iterator for neuron firing times
    ptr_w = weights.data_ptr<double>(); // init pointer to weights
    ptr_d = delays.data_ptr<double>(); // init pointer to delays
    ptr_s = sources.data_ptr<int>(); // init pointer to sources

    // For each neuron
    for(int l=0; l<L; l++)
    {
      if (*ptr_c > 0)
      {
        (*ptr_c)--;
        ptr_f++, ptr_c++, ptr_gc++;
        ptr_w += K, ptr_d += K, ptr_s += K;
        continue;
      }
      if (ptr_f->size() && (t - *(ptr_f->end() - 1) < Tr)) // if the neuron is still recovering from its last spike
      {
        ptr_f++, ptr_c++, ptr_gc++;
        ptr_w += K, ptr_d += K, ptr_s += K;
        continue;
      }
      z = d(gen); // continuous random potential noise
      for (int k=0; k<K; k++)
      {
        ptr_fs = firing_times[*ptr_s].rbegin();
        while (ptr_fs != firing_times[*ptr_s].rend())
        {
          tmp = t - *ptr_fs - *ptr_d;
          if (tmp > Tr) break;
          z += (*ptr_w) * impulse_resp(tmp, beta);
          ptr_fs++;
        }
        ptr_w++, ptr_d++, ptr_s++;
      }

      if (z >= theta) ptr_f->push_back(t); // neuron is spiking
      else *ptr_c = (int) ((theta - z) / *ptr_gc); // adaptive time step

      ptr_f++, ptr_c++, ptr_gc++;
    }
    // cout << t << endl;
    t+=DT;
  } while (t < max_t);

  return firing_times;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sim_cpp", &sim_cpp, "Simulation");
}