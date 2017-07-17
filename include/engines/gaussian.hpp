#ifndef GAUSSIAN_HEADER
#define GAUSSIAN_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class Gaussian : public OptionPricer {
  static DerivedRegister<Gaussian> reg;

public:
  Gaussian() {
    gpu = GPUContext("src/engines/gaussian.cl");
    pricerName = "Gaussian";
  }
  bool execute(int iterations);
};

#endif // GAUSSIAN_HEADER
