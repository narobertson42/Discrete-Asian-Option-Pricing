#include "engines/binomialCRR.hpp"

DerivedRegister<BinomialCRR> BinomialCRR::reg("BinomialCRR");

bool BinomialCRR::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();

    float dt = finalTime / steps;

    // Set Up Cox, Ross, Rubenstine Conditions
    float u = exp(double(sigma) * sqrt(double(dt)));
    float d = 1 / u;
    float p = (exp(double(interestRate) * double(dt)) - d) / (u - d);

    // Discrete Space
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

    kernel.setArg(0, callBuffer);
    uint conv_p = p * std::numeric_limits<uint>::max();
    ;
    kernel.setArg(1, conv_p);
    kernel.setArg(2, optionPrice);
    kernel.setArg(3, steps);
    srand(seed);
    kernel.setArg(4, (uint)rand());
    kernel.setArg(5, (uint)rand());
    kernel.setArg(6, stockPrice);
    kernel.setArg(7, u);
    kernel.setArg(8, d);

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

    calculateCallPrice(queue, callBuffer);

    timer->stop();

    calculateStatistics(queue, callBuffer);
  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
  return true;
}
