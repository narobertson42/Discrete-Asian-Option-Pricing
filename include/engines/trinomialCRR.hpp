#ifndef TRINOMIALCRR_HEADER
#define TRINOMIALCRR_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class TrinomialCRR : public OptionPricer {
  static DerivedRegister<TrinomialCRR> reg;

public:
  TrinomialCRR() {
    gpu = GPUContext("src/engines/trinomialCRR.cl");
    pricerName = "TrinomialCRR";
  }
  bool execute(int iterations);
};

#endif // TRINOMIALCRR_HEADER
