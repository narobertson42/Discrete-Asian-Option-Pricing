#include "engines/gaussian_CV.hpp"

DerivedRegister<Gaussian_CV> Gaussian_CV::reg("Gaussian_CV");

bool Gaussian_CV::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();

    float dt = finalTime / steps;

    cl::CommandQueue queue(gpu.context, gpu.device);

    // Kernel and Buffer Initialisation
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

    kernel.setArg(0, callBuffer);
    kernel.setArg(1, geometricCallBuffer);
    kernel.setArg(2, optionPrice);
    kernel.setArg(3, steps);
    srand(seed);
    kernel.setArg(4, (uint)rand());
    kernel.setArg(5, (uint)rand());
    kernel.setArg(6, stockPrice);
    kernel.setArg(7, interestRate);
    kernel.setArg(8, sigma);
    kernel.setArg(9, dt);

    // Iteration Space
    cl::NDRange offset(0);
    cl::NDRange globalSize;
    if (antithetic) {
      globalSize = cl::NDRange(iterations / 4);
    } else {
      globalSize = cl::NDRange(iterations / 2);
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

    unspecifiedVariable1 = c;
    unspecifiedVariable2 = geometricCallPrice - geometricCallPriceMC;

  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
  return true;
}

void Gaussian_CV::printHeaders() {
  OptionPricer::printHeaders();
  std::cout << ",c,Difference";
}

void Gaussian_CV::printCSV() {
  OptionPricer::printCSV();
  std::cout << "," << unspecifiedVariable1 << "," << unspecifiedVariable2;
}
