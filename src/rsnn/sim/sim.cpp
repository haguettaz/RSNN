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

vector<vector<double>> sim_cpp(double max_t, vector<vector<double>> firing_times, torch::Tensor sources, torch::Tensor delays, torch::Tensor weights, double Tr, double beta, double theta, double wb, double eta)
{
  // Counter for time steps, depends on the largest possible potential variation inbetween two consecutive time steps

  int L = sources.size(0);
  int K = sources.size(1);

  vector<int> counter(L, 0); // for time step adaptation
  vector<int>::iterator ptr_c;
  double t = 0; // simulation time
  double z; // neuron potential
  double tmp; // temporary time variable
  vector<vector<double>>::iterator ptr_f; // iterator for firing times
  vector<double>::reverse_iterator ptr_fs; // iterator for sources firing times
  double* ptr_w; // pointer to weights
  double* ptr_d; // pointer to delays
  int* ptr_s; // pointer to sources

  random_device rd{};
  mt19937 gen{rd()};
  normal_distribution<> d{0, eta};

  do
  {
    ptr_c = counter.begin();
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
        ptr_f++;
        ptr_c++;
        ptr_w += K;
        ptr_d += K;
        ptr_s += K;
        continue;
      }
      if (ptr_f->size() && (t - *(ptr_f->end() - 1) < Tr)) // if the neuron is still recovering from its last spike
      {
        ptr_f++;
        ptr_c++;
        ptr_w += K;
        ptr_d += K;
        ptr_s += K;
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
        ptr_w++;
        ptr_d++;
        ptr_s++;
      }

      if (z >= theta)
      {
        ptr_f->push_back(t); // neuron is spiking
        *ptr_c = 0; // reset time step counter
      } 
      else
      {
        // Note: the potential variation in-between two consecutive time steps is upper bounded by K * wb * exp(1) / beta
        // *ptr_c = (int) ((theta - z) * beta * 0.3678 / (DT * K * wb));
        // Note: the probability that all K inputs are firing at the same time is negligible and thus, we reduce the upper bound by a factor 10
        *ptr_c = (int) ((theta - z) * beta * 3.678 / (DT * K * wb));
      }
      ptr_f++;
      ptr_c++;
    }
    t+=DT;
  } while (t < max_t);

  return firing_times;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sim_cpp", &sim_cpp, "Simulation");
}