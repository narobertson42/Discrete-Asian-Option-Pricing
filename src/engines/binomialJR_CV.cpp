#include "engines/binomialJR_CV.hpp"

DerivedRegister<BinomialJR_CV> BinomialJR_CV::reg("BinomialJR_CV");

bool BinomialJR_CV::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();

    // Set Up Jarrow-Rudd Conditions
    float dt = double(finalTime) / double(steps);
    float u =
        exp(((double(interestRate) - (double(sigma) * double(sigma) / 2)) *
             double(dt)) +
            (double(sigma) * sqrt(double(dt))));
    float d =
        exp(((double(interestRate) - (double(sigma) * double(sigma) / 2)) *
             double(dt)) -
            (double(sigma) * sqrt(double(dt))));

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

    size_t treeBufferSize = 4 * 2;
    cl::Buffer shiftsBuffer(gpu.context, CL_MEM_READ_WRITE, treeBufferSize);

    float shifts[2] = {u, d};
    queue.enqueueWriteBuffer(shiftsBuffer, CL_TRUE, 0, treeBufferSize,
                             &shifts[0]);
    queue.enqueueBarrier();

    kernel.setArg(0, callBuffer);
    kernel.setArg(1, geometricCallBuffer);
    kernel.setArg(2, optionPrice);
    kernel.setArg(3, steps);
    srand(seed);
    kernel.setArg(4, (uint)rand());
    kernel.setArg(5, (uint)rand());
    kernel.setArg(6, stockPrice);
    kernel.setArg(7, shiftsBuffer);
    kernel.setArg(8, cl::__local(4 * 2));

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

    unspecifiedVariable1 = c;
    unspecifiedVariable2 = geometricCallPrice - geometricCallPriceMC;

  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
  return true;
}

void BinomialJR_CV::printHeaders() {
  OptionPricer::printHeaders();
  std::cout << ",c,Difference";
}

void BinomialJR_CV::printCSV() {
  OptionPricer::printCSV();
  std::cout << "," << unspecifiedVariable1 << "," << unspecifiedVariable2;
}
