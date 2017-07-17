#ifndef BINOMIALCRR_HEADER
#define BINOMIALCRR_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class BinomialCRR : public OptionPricer {
  static DerivedRegister<BinomialCRR> reg;

public:
  BinomialCRR() {
    gpu = GPUContext("src/engines/binomialCRR.cl");
    pricerName = "BinomialCRR";
  }
  bool execute(int iterations);
};

#endif // BINOMIALCRR_HEADER
