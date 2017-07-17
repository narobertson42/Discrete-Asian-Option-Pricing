#ifndef MULTINOMIAL_CV_HEADER
#define MULTINOMIAL_CV_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class Multinomial_CV : public OptionPricer {
  static DerivedRegister<Multinomial_CV> reg;

public:
  Multinomial_CV() {
    gpu = GPUContext("src/engines/multinomial_CV.cl");
    pricerName = "Multinomial_CV";
  }
  bool execute(int iterations);
  void printHeaders() override;
  void printCSV() override;
};

#endif // MULTINOMIAL_CV_HEADER
