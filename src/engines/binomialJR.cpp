#include "engines/binomialJR.hpp"

DerivedRegister<BinomialJR> BinomialJR::reg("BinomialJR");

bool BinomialJR::execute(int iterations) {
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

    size_t treeBufferSize = 4 * 2;
    cl::Buffer shiftsBuffer(gpu.context, CL_MEM_READ_WRITE, treeBufferSize);

    float shifts[2] = {u, d};
    queue.enqueueWriteBuffer(shiftsBuffer, CL_TRUE, 0, treeBufferSize,
                             &shifts[0]);
    queue.enqueueBarrier();

    kernel.setArg(0, callBuffer);
    kernel.setArg(1, optionPrice);
    kernel.setArg(2, steps);
    srand(seed);
    kernel.setArg(3, (uint)rand());
    kernel.setArg(4, (uint)rand());
    kernel.setArg(5, stockPrice);
    kernel.setArg(6, shiftsBuffer);
    kernel.setArg(7, cl::__local(4 * 2));

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
