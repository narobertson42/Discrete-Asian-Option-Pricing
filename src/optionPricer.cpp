#include "optionPricer.hpp"

int binomial(int n, int k) {
  int num, den;
  if (n < k) {
    return (0);
  } else {
    den = 1;
    num = 1;
    for (int i = 1; i <= k; i = i + 1) {
      den = den * i;
    }
    for (int j = n - k + 1; j <= n; j = j + 1) {
      num = num * j;
    }
    return (num / den);
  }
}

void OptionPricer::terminalFriendlyPrint() {
  std::cout << "\nEngine Name: " << pricerName;
  if (antithetic) {
    std::cout << "_Antithetic";
  }
  std::cout << std::endl;
  std::cout << "Stock Price: " << stockPrice << std::endl;
  std::cout << "Strike Price: " << optionPrice << std::endl;
  std::cout << "Interest Rate: " << interestRate << std::endl;
  std::cout << "Final Time: " << finalTime << std::endl;
  std::cout << "Volatility: " << sigma << std::endl;
  std::cout << "Steps: " << steps << std::endl;
  std::cout << "Iterations: " << iterationsExecuted << std::endl;
  std::cout << "Call Price: " << callPrice << std::endl;
  std::cout << "Mean: " << mean << std::endl;
  std::cout << "Variance: " << variance << std::endl;
  std::cout << "Approximate 99% Confidence Interval Size: "
            << (2.575 * sqrt(double(variance))) /
                   sqrt(double(iterationsExecuted))
            << std::endl;
  std::cout << "Execution Time: " << timer->format(6, "%w") << std::endl;
}

void OptionPricer::printHeaders() {
  std::cout << "Engine Name,Stock Price,Option Price,Interest,Final Time,"
               "Volatility,Steps,Iterations,Call Price,Mean,Variance,99% "
               "Confidence Interval,Time";
}

void OptionPricer::printCSV() {
  std::cout << "\n" << pricerName;
  if (antithetic) {
    std::cout << "_Antithetic";
  }
  std::cout << "," << stockPrice << "," << optionPrice << "," << interestRate
            << "," << finalTime << "," << sigma << "," << steps << ","
            << iterationsExecuted << "," << callPrice << "," << mean << ","
            << variance << ","
            << (2.575 * sqrt(double(variance))) /
                   sqrt(double(iterationsExecuted))
            << "," << timer->format(6, "%w");
}

double OptionPricer::calculateBufferMean(cl::CommandQueue &queue,
                                         cl::Buffer &buffer, int elements) {
  double bufferMean = 0;
  int divisionSize = 100;
  if (elements > divisionSize * 1000) {
    divisionSize = 1000;
  }

  if (elements % divisionSize != 0) {
    elements -= elements % divisionSize;
  }

  cl::Kernel meanKernel(gpu.program, "averageKernel");
  size_t meanBufferSize = 4 * elements / divisionSize;
  cl::NDRange offset(0);
  cl::NDRange meanGlobalSize(elements / divisionSize);
  cl::NDRange localSize = cl::NullRange;

  cl::Buffer compressedBuffer(gpu.context, CL_MEM_WRITE_ONLY, meanBufferSize);
  std::vector<float> compressedVector(elements / divisionSize, 0);

  meanKernel.setArg(0, buffer);
  meanKernel.setArg(1, compressedBuffer);
  meanKernel.setArg(2, divisionSize);
  meanKernel.setArg(3, elements);

  queue.enqueueNDRangeKernel(meanKernel, offset, meanGlobalSize, localSize);
  queue.enqueueBarrier();
  queue.enqueueReadBuffer(compressedBuffer, CL_TRUE, 0, meanBufferSize,
                          &compressedVector[0]);
  queue.enqueueBarrier();

  bufferMean = 0.0;
  for (auto it = compressedVector.begin(); it != compressedVector.end(); ++it) {
    bufferMean += *it;
  }
  bufferMean /= elements;

  return bufferMean;
}

double OptionPricer::calculateBufferVariance(cl::CommandQueue &queue,
                                             cl::Buffer &buffer, int elements,
                                             double meanIn) {
  double bufferVariance = 0.0;
  int divisionSize = 100;
  if (elements > divisionSize * 1000) {
    divisionSize = 1000;
  }

  if (elements % divisionSize != 0) {
    elements -= elements % divisionSize;
  }

  cl::Kernel meanKernel(gpu.program, "varianceKernel");
  size_t meanBufferSize = 4 * elements / divisionSize;
  cl::NDRange offset(0);
  cl::NDRange meanGlobalSize(elements / divisionSize);
  cl::NDRange localSize = cl::NullRange;

  cl::Buffer compressedBuffer(gpu.context, CL_MEM_WRITE_ONLY, meanBufferSize);
  std::vector<float> compressedVector(elements / divisionSize, 0);

  meanKernel.setArg(0, buffer);
  meanKernel.setArg(1, compressedBuffer);
  meanKernel.setArg(2, divisionSize);
  meanKernel.setArg(3, elements);
  meanKernel.setArg(4, float(meanIn));

  queue.enqueueNDRangeKernel(meanKernel, offset, meanGlobalSize, localSize);
  queue.enqueueBarrier();
  queue.enqueueReadBuffer(compressedBuffer, CL_TRUE, 0, meanBufferSize,
                          &compressedVector[0]);
  queue.enqueueBarrier();

  bufferVariance = 0;
  for (auto it = compressedVector.begin(); it != compressedVector.end(); ++it) {
    bufferVariance += *it;
  }
  bufferVariance /= (elements - 1);

  return bufferVariance;
}

double OptionPricer::calculateBufferVarianceAntithetic(cl::CommandQueue &queue,
                                                       cl::Buffer &buffer,
                                                       int elements) {
  double bufferVariance = 0.0;

  size_t subBufferSize = 4 * elements / 2;
  cl_buffer_region regionA = {0, subBufferSize};
  cl::Buffer bufferA = buffer.createSubBuffer(
      CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &regionA);
  cl_buffer_region regionB = {subBufferSize, subBufferSize};
  cl::Buffer bufferB = buffer.createSubBuffer(
      CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &regionB);

  double meanA = calculateBufferMean(queue, bufferA, elements / 2);
  double meanB = calculateBufferMean(queue, bufferB, elements / 2);
  double varianceA =
      calculateBufferVariance(queue, bufferA, elements / 2, meanA);
  double varianceB =
      calculateBufferVariance(queue, bufferB, elements / 2, meanB);
  double covariance = calculateBufferCovariance(queue, bufferA, bufferB,
                                                elements / 2, meanA, meanB);
  mean = 0.5 * meanA + 0.5 * meanB;
  bufferVariance = 0.5 * (varianceA + varianceB + 2 * covariance);
  return bufferVariance;
}

double OptionPricer::calculateBufferCovariance(cl::CommandQueue &queue,
                                               cl::Buffer &buffer1,
                                               cl::Buffer &buffer2,
                                               int elements, double meanIn1,
                                               double meanIn2) {
  double bufferCovariance = 0;
  int divisionSize = 100;
  if (elements > divisionSize * 1000) {
    divisionSize = 1000;
  }

  if (elements % divisionSize != 0) {
    elements -= elements % divisionSize;
  }

  cl::Kernel meanKernel(gpu.program, "covarianceKernel");
  size_t meanBufferSize = 4 * elements / divisionSize;
  cl::NDRange offset(0);
  cl::NDRange meanGlobalSize(elements / divisionSize);
  cl::NDRange localSize = cl::NullRange;

  cl::Buffer resultsBuffer(gpu.context, CL_MEM_WRITE_ONLY, meanBufferSize);
  std::vector<float> compressedVector(elements / divisionSize, 0);

  meanKernel.setArg(0, buffer1);
  meanKernel.setArg(1, buffer2);
  meanKernel.setArg(2, resultsBuffer);
  meanKernel.setArg(3, divisionSize);
  meanKernel.setArg(4, elements);
  meanKernel.setArg(5, float(meanIn1));
  meanKernel.setArg(6, float(meanIn2));

  queue.enqueueNDRangeKernel(meanKernel, offset, meanGlobalSize, localSize);
  queue.enqueueBarrier();
  queue.enqueueReadBuffer(resultsBuffer, CL_TRUE, 0, meanBufferSize,
                          &compressedVector[0]);
  queue.enqueueBarrier();

  bufferCovariance = 0;
  for (auto it = compressedVector.begin(); it != compressedVector.end(); ++it) {
    bufferCovariance += *it;
  }
  bufferCovariance /= (elements - 1);

  return bufferCovariance;
}

double OptionPricer::calculateBufferCovarianceAntithetic(
    cl::CommandQueue &queue, cl::Buffer &buffer1, cl::Buffer &buffer2,
    int elements) {
  double bufferCovariance = 0;
  int divisionSize = 100;
  if (elements > divisionSize * 1000) {
    divisionSize = 1000;
  }

  if (elements % divisionSize != 0) {
    elements -= elements % divisionSize;
  }

  double bufferVariance = 0.0;

  size_t subBufferSize = 4 * elements / 2;
  cl_buffer_region region1 = {0, subBufferSize};
  cl_buffer_region region2 = {subBufferSize, subBufferSize};

  cl::Buffer bufferA1 = buffer1.createSubBuffer(
      CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region1);
  cl::Buffer bufferA2 = buffer1.createSubBuffer(
      CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region2);
  cl::Buffer bufferB1 = buffer2.createSubBuffer(
      CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region1);
  cl::Buffer bufferB2 = buffer2.createSubBuffer(
      CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region2);

  double meanA1 = calculateBufferMean(queue, bufferA1, elements / 2);
  double meanA2 = calculateBufferMean(queue, bufferA2, elements / 2);
  double meanB1 = calculateBufferMean(queue, bufferB1, elements / 2);
  double meanB2 = calculateBufferMean(queue, bufferB2, elements / 2);

  double covariance1 = calculateBufferCovariance(queue, bufferA1, bufferB1,
                                                 elements / 2, meanA1, meanB1);
  double covariance2 = calculateBufferCovariance(queue, bufferA1, bufferB2,
                                                 elements / 2, meanA1, meanB2);
  double covariance3 = calculateBufferCovariance(queue, bufferA2, bufferB1,
                                                 elements / 2, meanA2, meanB1);
  double covariance4 = calculateBufferCovariance(queue, bufferA2, bufferB2,
                                                 elements / 2, meanA2, meanB2);

  bufferCovariance =
      covariance1 + covariance2 + covariance3 + covariance4; // by Bilinearity

  std::cout << "Covariance1: " << covariance1 << std::endl;
  std::cout << "Covariance2: " << covariance2 << std::endl;
  std::cout << "Covariance3: " << covariance3 << std::endl;
  std::cout << "Covariance4: " << covariance4 << std::endl;

  return bufferCovariance;
}

double OptionPricer::calculateCallPrice(cl::CommandQueue &queue,
                                        cl::Buffer &callBuffer) {
  mean = calculateBufferMean(queue, callBuffer, iterationsExecuted);
  callPrice = mean * exp(-interestRate * finalTime);
  return callPrice;
}

double OptionPricer::calculateControlVariateCallPrice(
    cl::CommandQueue &queue, cl::Buffer &callBuffer,
    cl::Buffer &geometricCallBuffer, int elements) {
  double geometricCallPrice = calculateGeometricCallPrice();
  double geometricCallPriceMC =
      calculateBufferMean(queue, geometricCallBuffer, elements);
  double asianCallPrice = calculateBufferMean(queue, callBuffer, elements);
  double covariance =
      calculateBufferCovariance(queue, geometricCallBuffer, callBuffer,
                                elements, geometricCallPriceMC, asianCallPrice);
  double varianceGeometric = calculateBufferVariance(
      queue, geometricCallBuffer, elements, geometricCallPriceMC);
  double c = covariance / varianceGeometric;
  callPrice =
      (asianCallPrice + c * (geometricCallPrice - geometricCallPriceMC)) *
      exp(-interestRate * (finalTime));
  unspecifiedVariable1 = c;
  unspecifiedVariable2 = geometricCallPrice - geometricCallPriceMC;
  unspecifiedVariable3 = covariance;
  return callPrice;
}

void OptionPricer::calculateStatistics(cl::CommandQueue &queue,
                                       cl::Buffer &buffer) {
  int iterations = iterationsExecuted;
  mean = calculateBufferMean(queue, buffer, iterations);
  if (antithetic) {
    variance = calculateBufferVarianceAntithetic(queue, buffer, iterations);
  } else {
    variance = calculateBufferVariance(queue, buffer, iterations, mean);
  }
}

// Not entirely dynamic testing, but gives peace of mind
void OptionPricer::testModelCorrectness() {
  stockPrice = 100.0;             // Stock Price
  optionPrice = 110.0;            // Option Price
  interestRate = 0.15;            // Interest Rate
  finalTime = 1.0;                // Final Time
  sigma = 0.3;                    // Volatility
  steps = 1000;                   // Steps
  int N = 1000000;                // Iterations
  double expectedValue = 5.73007; // This is only approximate true value

  execute(N);
  float standardDeviation = sqrt(variance);

  // 99.9%
  float boundaryLeniancy =
      (3.29 * sqrt(double(variance))) / sqrt(double(iterationsExecuted)) +
      0.0002;
  std::string antitheticString = "";
  if (antithetic) {
    antitheticString = "_Antithetic";
  }

  if (expectedValue - boundaryLeniancy < callPrice &&
      callPrice < expectedValue + boundaryLeniancy) {
    std::cout << "PASS: " << pricerName << antitheticString << std::endl;
  } else {
    std::cout << "FAIL: " << pricerName << antitheticString << std::endl;
    terminalFriendlyPrint();
  }
}

double OptionPricer::calculateEuropeanCallPrice() {
  double d1 = (log(stockPrice / optionPrice) +
               (interestRate + sigma * sigma / 2.) * (finalTime)) /
              (sigma * sqrt(finalTime));

  double d2 = (log(stockPrice / optionPrice) +
               (interestRate - sigma * sigma / 2.) * (finalTime)) /
              (sigma * sqrt(finalTime));

  double europeanCallPrice =
      normalDistribution(d1) * stockPrice -
      normalDistribution(d2) * optionPrice * exp(-interestRate * (finalTime));

  return europeanCallPrice * exp(interestRate * (finalTime));
}

double OptionPricer::calculateGeometricCallPrice() {
  double dstar = 0.5 * (interestRate - sigma * sigma / 6.0) * finalTime;

  double d1 = (log(stockPrice / optionPrice) +
               0.5 * (interestRate + sigma * sigma / 6.0) * finalTime) /
              (sigma * sqrt(finalTime / 3.0));

  double d2 = d1 - sigma * sqrt(finalTime / 3.0);

  return exp(dstar) * normalDistribution(d1) * stockPrice -
         normalDistribution(d2) * optionPrice;
}

double normalDistribution(double x) {
  static const double RT2PI = sqrt(4.0 * acos(0.0));
  static const double SPLIT = 10. / sqrt(2);
  static const double a[] = {
      220.206867912376, 221.213596169931,  112.079291497871,    33.912866078383,
      6.37396220353165, 0.700383064443688, 3.52624965998911e-02};
  static const double b[] = {440.413735824752, 793.826512519948,
                             637.333633378831, 296.564248779674,
                             86.7807322029461, 16.064177579207,
                             1.75566716318264, 8.83883476483184e-02};

  const double z = fabs(x);
  double Nz = 0.0;

  // if z outside these limits then value effectively 0 or 1 for machine
  // precision
  if (z <= 37.0) {
    // NDash = N'(z) * sqrt{2\pi}
    const double NDash = exp(-z * z / 2.0) / RT2PI;
    if (z < SPLIT) {
      const double Pz =
          (((((a[6] * z + a[5]) * z + a[4]) * z + a[3]) * z + a[2]) * z +
           a[1]) *
              z +
          a[0];
      const double Qz =
          ((((((b[7] * z + b[6]) * z + b[5]) * z + b[4]) * z + b[3]) * z +
            b[2]) *
               z +
           b[1]) *
              z +
          b[0];
      Nz = RT2PI * NDash * Pz / Qz;
    } else {
      const double F4z =
          z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 13.0 / 20.0))));
      Nz = NDash / F4z;
    }
  }
  return x >= 0.0 ? 1 - Nz : Nz;
}

double rationalApproximation(double t) {
  // Abramowitz and Stegun formula 26.2.23.
  // The absolute value of the error should be less than 4.5 e-4.
  double c[] = {2.515517, 0.802853, 0.010328};
  double d[] = {1.432788, 0.189269, 0.001308};
  return t -
         ((c[2] * t + c[1]) * t + c[0]) /
             (((d[2] * t + d[1]) * t + d[0]) * t + 1.0);
}

// See header for source
double normalCDFInverse(double p) {
  if (p <= 0.0 || p >= 1.0) {
    std::stringstream os;
    os << "Invalid input argument (" << p
       << "); must be larger than 0 but less than 1.";
    throw std::invalid_argument(os.str());
  }
  if (p < 0.5) {
    // F^-1(p) = - G^-1(p)
    return -rationalApproximation(sqrt(-2.0 * log(p)));
  } else {
    // F^-1(p) = G^-1(1-p)
    return rationalApproximation(sqrt(-2.0 * log(1 - p)));
  }
}
