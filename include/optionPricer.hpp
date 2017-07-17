#ifndef OPTION_PRICER_HEADER
#define OPTION_PRICER_HEADER

#include "GPUContext.hpp"
#include <algorithm>
#include <boost/timer/timer.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class OptionPricer {
public:
  OptionPricer() { timer = new boost::timer::cpu_timer(); };
  OptionPricer(float stockPriceIn, float optionPriceIn, float interestRateIn,
               float finalTimeIn, float sigmaIn, int stepsIn, int seedIn,
               bool antitheticIn) {
    stockPrice = stockPriceIn;
    optionPrice = optionPriceIn;
    interestRate = interestRateIn;
    finalTime = finalTimeIn;
    sigma = sigmaIn;
    steps = stepsIn;
    seed = seedIn;
    timer = new boost::timer::cpu_timer();
    mean = 0;
    variance = 0;
    antithetic = antitheticIn;
  };

  void UpdateParameters(float stockPriceIn, float optionPriceIn,
                        float interestRateIn, float finalTimeIn, float sigmaIn,
                        int stepsIn, int seedIn, bool antitheticIn) {
    stockPrice = stockPriceIn;
    optionPrice = optionPriceIn;
    interestRate = interestRateIn;
    finalTime = finalTimeIn;
    sigma = sigmaIn;
    steps = stepsIn;
    seed = seedIn;
    timer = new boost::timer::cpu_timer();
    mean = 0;
    variance = 0;
    antithetic = antitheticIn;
  };

  virtual bool execute(int iterations) = 0;
  void terminalFriendlyPrint();
  virtual void printHeaders();
  virtual void printCSV();
  double calculateCallPrice(cl::CommandQueue &queue, cl::Buffer &callbuffer);
  double calculateControlVariateCallPrice(cl::CommandQueue &queue,
                                          cl::Buffer &callBuffer,
                                          cl::Buffer &geometricCallBuffer,
                                          int elements);
  double calculateBufferMean(cl::CommandQueue &queue, cl::Buffer &stockBuffer,
                             int elements);
  double calculateBufferVariance(cl::CommandQueue &queue, cl::Buffer &buffer,
                                 int elements, double meanIn);
  double calculateBufferVarianceAntithetic(cl::CommandQueue &queue,
                                           cl::Buffer &buffer, int elements);
  double calculateBufferCovariance(cl::CommandQueue &queue, cl::Buffer &buffer1,
                                   cl::Buffer &buffer2, int elements,
                                   double meanIn1, double meanIn2);
  double calculateBufferCovarianceAntithetic(cl::CommandQueue &queue,
                                             cl::Buffer &buffer1,
                                             cl::Buffer &buffer2, int elements);
  void calculateStatistics(cl::CommandQueue &queue, cl::Buffer &buffer);
  double calculateEuropeanCallPrice();
  double calculateGeometricCallPrice();
  void testModelCorrectness();

protected:
  float stockPrice = 100;
  float optionPrice = 110;
  float interestRate = 0.1;
  float finalTime = 1;
  float sigma = 0.3;
  int steps = 1000;
  int seed = 42;
  int iterationsExecuted = 1000;

  int walksPerKernel = 1;
  bool statistics = true;
  bool antithetic = false;

  double callPrice = 0;

  double mean = 0;
  double variance = 0;

  // XXX Hack to print out additional information, need to refactor out
  double unspecifiedVariable1 = 0.0;
  double unspecifiedVariable2 = 0.0;
  double unspecifiedVariable3 = 0.0;

  GPUContext gpu;

  boost::timer::cpu_timer *timer;

  std::string pricerName;
};

template <typename T> OptionPricer *createT() { return new T; }

// Necessary mathematical functions based on
// https://www.johndcook.com/blog/cpp_phi_inverse/
double normalDistribution(double x);
int binomial(int n, int k);
double normalCDFInverse(double p);
double rationalApproximation(double t);

#endif // OPTION_PRICER_HEADER
