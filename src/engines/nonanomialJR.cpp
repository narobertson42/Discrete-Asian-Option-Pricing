#include "engines/nonanomialJR.hpp"

DerivedRegister<NonanomialJR> NonanomialJR::reg("NonanomialJR");

bool NonanomialJR::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();
    float dt = finalTime / steps;

    double dt2 = double(finalTime) / (double(steps) * 8.);

    // Set Jarrow-Rudd Conditions
    double u =
        exp(((double(interestRate) - (double(sigma) * double(sigma) / 2)) *
             double(dt2)) +
            (double(sigma) * sqrt(double(dt2))));
    double d =
        exp(((double(interestRate) - (double(sigma) * double(sigma) / 2)) *
             double(dt2)) -
            (double(sigma) * sqrt(double(dt2))));

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

    uint treeSize = 256;
    size_t treeBufferSize = 4 * 256;
    cl::Buffer shiftsBuffer(gpu.context, CL_MEM_READ_WRITE, treeBufferSize);
    float shifts[256];

    uint i = 0;
    for (uint j = 0; j <= 8; j++) {
      uint fill = binomial(8, j);
      for (uint k = i; k < i + fill; k++) {
        shifts[k] = std::pow(u, j) * std::pow(d, 8 - j);
      }
      i = i + fill;
    }

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
    kernel.setArg(7, cl::__local(4 * 256));

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
