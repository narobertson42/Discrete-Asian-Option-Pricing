#ifndef BINOMIALCRR_CV_HEADER
#define BINOMIALCRR_CV_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class BinomialCRR_CV : public OptionPricer {
  static DerivedRegister<BinomialCRR_CV> reg;

public:
  BinomialCRR_CV() {
    gpu = GPUContext("src/engines/binomialCRR_CV.cl");
    pricerName = "BinomialCRR_CV";
  }
  bool execute(int iterations);
  void printHeaders() override;
  void printCSV() override;
};

#endif // BINOMIALCRR_CV_HEADER
