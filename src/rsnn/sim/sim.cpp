#include <torch/extension.h>

#include <vector>

void sim_cpp(torch::Tensor sources, torch::Tensor delays, torch::Tensor weights, double max_t, double dt)
{
  double t = 0;
  do
  {
    t+=dt;
  } while (t < max_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sim_cpp", &sim_cpp, "Simulation");
}