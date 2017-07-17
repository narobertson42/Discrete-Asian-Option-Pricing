#include "engines/binomialCRR_CV.hpp"

DerivedRegister<BinomialCRR_CV> BinomialCRR_CV::reg("BinomialCRR_CV");

bool BinomialCRR_CV::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();

    // Binomial branch conditions determined using Cox, Ross, and Rubinstein
    // model
    double dt = double(finalTime) / double(steps);
    double u = exp(double(sigma) * sqrt(double(dt)));
    double d = 1 / u;
    double p = (exp(double(interestRate) * double(dt)) - d) / (u - d);

    cl::CommandQueue queue(gpu.context, gpu.device);
    cl::Kernel kernel;

    if (antithetic) {
      kernel = cl::Kernel(gpu.program, "discreteMonteCarloKernelAntithetic");
    } else {
      kernel = cl::Kernel(gpu.program, "discreteMonteCarloKernel");
    }

    size_t valueBufferSize = 4 * iterations;
    cl::Buffer callBuffer(gpu.context, CL_MEM_READ_WRITE, valueBufferSize);
    cl::Buffer geometricCallBuffer(gpu.context, CL_MEM_READ_WRITE,
                                   valueBufferSize);

    // Setting kernel arguments
    kernel.setArg(0, callBuffer);
    kernel.setArg(1, geometricCallBuffer);
    kernel.setArg(2, optionPrice);
    kernel.setArg(3, steps);
    srand(seed);
    kernel.setArg(4, (uint)rand());
    kernel.setArg(5, (uint)rand());
    kernel.setArg(6, stockPrice);
    kernel.setArg(7, (float)u);
    kernel.setArg(8, (float)d);
    uint conv_p = p * std::numeric_limits<uint>::max();
    kernel.setArg(9, conv_p);

    // Iteration Space
    cl::NDRange offset(0);
    cl::NDRange globalSize;
    if (antithetic) {
      globalSize = cl::NDRange(iterations / 2);
    } else {
      globalSize = cl::NDRange(iterations);
    }
    cl::NDRange localSize = cl::NullRange;

    queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
    queue.enqueueBarrier();

    double geometricCallPrice = calculateGeometricCallPrice();
    double geometricCallPriceMC =
        calculateBufferMean(queue, geometricCallBuffer, iterations);
    double asianCallPrice = calculateBufferMean(queue, callBuffer, iterations);
    double covariance = calculateBufferCovariance(
        queue, geometricCallBuffer, callBuffer, iterations,
        geometricCallPriceMC, asianCallPrice);
    double varianceGeometric = calculateBufferVariance(
        queue, geometricCallBuffer, iterations, geometricCallPriceMC);
    double c = covariance / varianceGeometric;
    callPrice =
        (asianCallPrice + c * (geometricCallPrice - geometricCallPriceMC)) *
        exp(-interestRate * (finalTime));

    timer->stop();

    mean = (asianCallPrice + c * (geometricCallPrice - geometricCallPriceMC));
    variance = calculateBufferVariance(queue, callBuffer, iterations, mean) -
               (covariance * covariance) / varianceGeometric;

  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
  return true;
}

void BinomialCRR_CV::printHeaders() {
  OptionPricer::printHeaders();
  std::cout << ",c,Difference";
}

void BinomialCRR_CV::printCSV() {
  OptionPricer::printCSV();
  std::cout << "," << unspecifiedVariable1 << "," << unspecifiedVariable2;
}
