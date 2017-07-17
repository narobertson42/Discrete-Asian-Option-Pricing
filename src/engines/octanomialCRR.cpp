#include "engines/octanomialCRR.hpp"

DerivedRegister<OctanomialCRR> OctanomialCRR::reg("OctanomialCRR");

bool OctanomialCRR::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();
    float dt = finalTime / steps;

    double dt2 = double(finalTime) / (double(steps) * 7.);

    double u = exp(double(sigma) * sqrt(double(dt2)));
    double d = 1 / u;
    double p = (exp(double(interestRate) * double(dt2)) - d) / (u - d);

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

    uint treeSize = 8;
    size_t treeBufferSize = 4 * 8;
    cl::Buffer shiftsBuffer(gpu.context, CL_MEM_READ_WRITE, treeBufferSize);
    cl::Buffer probabilityBuffer(gpu.context, CL_MEM_READ_WRITE,
                                 treeBufferSize);
    float shifts[8];
    uint probabilities[8];

    uint i = 0;
    for (uint j = 0; j < 8; j++) {
      probabilities[j] = binomial(7, j) * std::pow(p, j) *
                         std::pow((1 - p), 7 - j) *
                         std::numeric_limits<uint>::max();
      shifts[j] = std::pow(u, j) * std::pow(d, 7 - j);
    }

    for (uint j = 1; j < 8; j++) {
      probabilities[j] += probabilities[j - 1];
    }

    probabilities[7] = std::numeric_limits<uint>::max();

    queue.enqueueWriteBuffer(shiftsBuffer, CL_TRUE, 0, treeBufferSize,
                             &shifts[0]);
    queue.enqueueWriteBuffer(probabilityBuffer, CL_TRUE, 0, treeBufferSize,
                             &probabilities[0]);
    queue.enqueueBarrier();

    kernel.setArg(0, callBuffer);
    kernel.setArg(1, optionPrice);
    kernel.setArg(2, steps);
    srand(seed);
    kernel.setArg(3, (uint)rand());
    kernel.setArg(4, (uint)rand());
    kernel.setArg(5, stockPrice);
    kernel.setArg(6, shiftsBuffer);
    kernel.setArg(7, cl::__local(4 * 8));
    kernel.setArg(8, probabilityBuffer);
    kernel.setArg(9, cl::__local(4 * 8));

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
