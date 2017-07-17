#ifndef NONANOMIALJR_HEADER
#define NONANOMIALJR_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class NonanomialJR : public OptionPricer {
  static DerivedRegister<NonanomialJR> reg;

public:
  NonanomialJR() {
    gpu = GPUContext("src/engines/nonanomialJR.cl");
    pricerName = "NonanomialJR";
  }
  bool execute(int iterations);
};

#endif // NONANOMIALJR_HEADER
