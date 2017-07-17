#ifndef BINOMIALJR_CV_HEADER
#define BINOMIALJR_CV_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class BinomialJR_CV : public OptionPricer {
  static DerivedRegister<BinomialJR_CV> reg;

public:
  BinomialJR_CV() {
    gpu = GPUContext("src/engines/binomialJR_CV.cl");
    pricerName = "BinomialJR_CV";
  }
  bool execute(int iterations);
  void printHeaders() override;
  void printCSV() override;
};

#endif // BINOMIALJR_CV_HEADER
