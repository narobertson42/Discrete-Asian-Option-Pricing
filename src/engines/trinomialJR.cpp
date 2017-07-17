#include "engines/trinomialJR.hpp"

DerivedRegister<TrinomialJR> TrinomialJR::reg("TrinomialJR");

bool TrinomialJR::execute(int iterations) {
  iterationsExecuted = iterations;
  try {
    timer->start();
    float dt = finalTime / steps;
    float dt2 = double(finalTime) / double(steps);

    // Set Jarrow-Rudd Conditions
    double u = exp(
        ((double(interestRate) - (double(sigma) * double(sigma) / double(2))) *
         double(dt2)) +
        (double(sigma) * sqrt(double(2) * double(dt2))));
    double d = exp(
        ((double(interestRate) - (double(sigma) * double(sigma) / double(2))) *
         double(dt2)) -
        (double(sigma) * sqrt(double(2) * double(dt2))));

    float u2 = u;
    float d2 = d;
    float ud = exp(
        (double(interestRate) - (double(sigma) * double(sigma) / double(2))) *
        double(dt));
    float ud2 = ud;

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
    uint treeSize = 4;
    size_t treeBufferSize = 4 * 4;
    cl::Buffer shiftsBuffer(gpu.context, CL_MEM_READ_WRITE, treeBufferSize);
    float shifts[4] = {u2, ud, ud, d2};
    // float shifts[3] = {u2, ud, d2}; //Popcount Method
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
    kernel.setArg(7, cl::__local(4 * 4));

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
