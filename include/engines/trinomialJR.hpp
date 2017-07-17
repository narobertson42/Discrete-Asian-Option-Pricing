#ifndef TRINOMIALJR_HEADER
#define TRINOMIALJR_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class TrinomialJR : public OptionPricer {
  static DerivedRegister<TrinomialJR> reg;

public:
  TrinomialJR() {
    gpu = GPUContext("src/engines/trinomialJR.cl");
    pricerName = "TrinomialJR";
  }
  bool execute(int iterations);
};

#endif // TRINOMIALJR_HEADER
