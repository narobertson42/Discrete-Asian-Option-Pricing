#ifndef BINOMIALJR_HEADER
#define BINOMIALJR_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class BinomialJR : public OptionPricer {
  static DerivedRegister<BinomialJR> reg;

public:
  BinomialJR() {
    gpu = GPUContext("src/engines/binomialJR.cl");
    pricerName = "BinomialJR";
  }
  bool execute(int iterations);
};

#endif // BINOMIALJR_HEADER
