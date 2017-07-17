#ifndef BINOMIALCRR_MEMORY_HEADER
#define BINOMIALCRR_MEMORY_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class BinomialCRR_Memory : public OptionPricer {
  static DerivedRegister<BinomialCRR_Memory> reg;

public:
  BinomialCRR_Memory() {
    gpu = GPUContext("src/engines/binomialCRR_Memory.cl");
    pricerName = "BinomialCRR_Memory";
  }
  bool execute(int iterations);
  void printHeaders() override;
  void printCSV() override;
};

#endif // BINOMIALCRR_MEMORY_HEADER
