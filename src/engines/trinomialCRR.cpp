#include "engines/trinomialCRR.hpp"

DerivedRegister<TrinomialCRR> TrinomialCRR::reg("TrinomialCRR");

bool TrinomialCRR::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();

    float dt = finalTime / steps;

    // Set Up Cox, Ross, Rubenstine Conditions
    float u = exp(sigma * sqrt(2 * dt));
    float d = 1 / u;
    float pu = (exp(interestRate * dt / 2) - exp(-sigma * sqrt(dt / 2))) /
               (exp(sigma * sqrt(dt / 2)) - exp(-sigma * sqrt(dt / 2)));
    pu = pu * pu;
    float pd = (exp(sigma * sqrt(dt / 2)) - exp(interestRate * dt / 2)) /
               (exp(sigma * sqrt(dt / 2)) - exp(-sigma * sqrt(dt / 2)));
    pd = pd * pd;
    float ps = 1 - pu - pd;

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
    uint conv_pu = pu * std::numeric_limits<uint>::max();
    uint conv_ps = conv_pu + ps * std::numeric_limits<uint>::max();
    kernel.setArg(1, conv_pu);
    kernel.setArg(2, conv_ps);
    kernel.setArg(3, optionPrice);
    kernel.setArg(4, steps);
    srand(seed);
    kernel.setArg(5, (uint)rand());
    kernel.setArg(6, (uint)rand());
    kernel.setArg(7, stockPrice);
    kernel.setArg(8, u);
    kernel.setArg(9, d);

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
