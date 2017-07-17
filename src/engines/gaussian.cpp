#include "engines/gaussian.hpp"

DerivedRegister<Gaussian> Gaussian::reg("Gaussian");

bool Gaussian::execute(int iterations) {
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

    kernel.setArg(0, callBuffer);
    kernel.setArg(1, optionPrice);
    kernel.setArg(2, steps);
    srand(seed);
    kernel.setArg(3, (uint)rand());
    kernel.setArg(4, (uint)rand());
    kernel.setArg(5, stockPrice);
    kernel.setArg(6, interestRate);
    kernel.setArg(7, sigma);
    kernel.setArg(8, dt);

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

    calculateCallPrice(queue, callBuffer);

    timer->stop();

    calculateStatistics(queue, callBuffer);

  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
  return true;
}
