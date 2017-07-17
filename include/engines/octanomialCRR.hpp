#ifndef OCTANOMIALCRR_HEADER
#define OCTANOMIALCRR_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class OctanomialCRR : public OptionPricer {
  static DerivedRegister<OctanomialCRR> reg;

public:
  OctanomialCRR() {
    gpu = GPUContext("src/engines/octanomialCRR.cl");
    pricerName = "OctanomialCRR";
  }
  bool execute(int iterations);
};

#endif // OCTANOMIALCRR_HEADER
