#include "engines/multinomial.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

DerivedRegister<Multinomial> Multinomial::reg("Multinomial");

bool Multinomial::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();
    float dt = finalTime / steps;

    uint branches = 64;
    double probabilityValues[64];
    float shifts[64];
    double shiftsMean = 0;
    double shiftsStandardDeviation = 0;
    for (uint i = 0; i < 64; i++) {
      probabilityValues[i] =
          (i * (1.0 / branches) + (i + 1) * (1.0 / branches)) / 2.0;
      shifts[i] = normalCDFInverse(probabilityValues[i]);
      shiftsMean += shifts[i];
    }

    shiftsMean = shiftsMean / 64.0;
    for (uint i = 0; i < 64; i++) {
      shiftsStandardDeviation += (shifts[i] - mean) * (shifts[i] - mean);
    }
    shiftsStandardDeviation = sqrt(shiftsStandardDeviation / 64.0);

    for (uint i = 0; i < 64; i++) {
      shifts[i] = shifts[i] / shiftsStandardDeviation;
      shifts[i] = exp((interestRate - 0.5 * sigma * sigma) * dt +
                      sigma * sqrt(dt) * shifts[i]);
    }

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

    size_t treeBufferSize = 4 * branches;
    cl::Buffer shiftsBuffer(gpu.context, CL_MEM_READ_WRITE, treeBufferSize);

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
    kernel.setArg(7, cl::__local(4 * branches));

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
