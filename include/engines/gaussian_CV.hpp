#ifndef GAUSSIAN_CV_HEADER
#define GAUSSIAN_CV_HEADER

#include "GPUContext.hpp"
#include "optionPricer.hpp"
#include "pricerRegistrar.hpp"

class Gaussian_CV : public OptionPricer {
  static DerivedRegister<Gaussian_CV> reg;

public:
  Gaussian_CV() {
    gpu = GPUContext("src/engines/gaussian_CV.cl");
    pricerName = "Gaussian_CV";
  }
  bool execute(int iterations);
  void printHeaders() override;
  void printCSV() override;
};

#endif // GAUSSIAN_CV_HEADER
