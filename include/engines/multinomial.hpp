#ifndef MULTINOMIAL_HEADER
#define MULTINOMIAL_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class Multinomial : public OptionPricer {
  static DerivedRegister<Multinomial> reg;

public:
  Multinomial() {
    gpu = GPUContext("src/engines/multinomial.cl");
    pricerName = "Multinomial";
  }
  bool execute(int iterations);
};

#endif // MULTINOMIAL_HEADER
