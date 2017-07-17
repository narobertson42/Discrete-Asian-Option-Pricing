#include "engines/binomialCRR_Memory.hpp"

DerivedRegister<BinomialCRR_Memory>
    BinomialCRR_Memory::reg("BinomialCRR_Memory");

bool BinomialCRR_Memory::execute(int iterations) {

  walksPerKernel = 1;
  if (getenv("WALKS_PER_KERNEL")) {
    walksPerKernel = atoi(getenv("WALKS_PER_KERNEL"));
  }

  if (iterations % walksPerKernel != 0) {
    iterations -= iterations % walksPerKernel;
  }
  iterationsExecuted = iterations;

  try {
    timer->start();

    // Set Up Cox, Ross, Rubinstein Conditions
    double dt = double(finalTime) / double(steps);
    double u = exp(double(sigma) * sqrt(double(dt)));
    double d = 1 / u;
    double p = (exp(double(interestRate) * double(dt)) - d) / (u - d);

    // Discrete stock price space calculations
    int numberOfDiscretisations = (2 * steps) + 1;
    int midPoint = steps;

    std::vector<double> doubleDiscreteValues(numberOfDiscretisations,
                                             stockPrice);
    std::vector<float> discreteValues(numberOfDiscretisations, 0);

    // Upper Tree
    for (int i = midPoint - 1; i >= 0; i--) {
      doubleDiscreteValues[i] = doubleDiscreteValues[i + 1] * d;
    }

    // Lower Tree
    for (int i = midPoint + 1; i < numberOfDiscretisations; i++) {
      doubleDiscreteValues[i] = doubleDiscreteValues[i - 1] * u;
    }

    for (int i = 0; i < numberOfDiscretisations; i++) {
      discreteValues[i] = doubleDiscreteValues[i];
    }

    cl::CommandQueue queue(gpu.context, gpu.device);

    // Kernel and Buffer Initialisation
    cl::Kernel kernel;
    if (antithetic) {
      kernel = cl::Kernel(gpu.program, "discreteMonteCarloKernelAntithetic");
    } else {
      kernel = cl::Kernel(gpu.program, "discreteMonteCarloKernel");
    }
    size_t discreteBufferSize = 4 * numberOfDiscretisations;
    size_t valueBufferSize = 4 * iterations / walksPerKernel;

    cl::Buffer discreteBuffer(gpu.context, CL_MEM_READ_WRITE,
                              discreteBufferSize);
    cl::Buffer callBuffer(gpu.context, CL_MEM_READ_WRITE, valueBufferSize);

    queue.enqueueWriteBuffer(discreteBuffer, CL_TRUE, 0, discreteBufferSize,
                             &discreteValues[0]);
    queue.enqueueBarrier();

    kernel.setArg(0, discreteBuffer);
    kernel.setArg(1, cl::__local(4 * numberOfDiscretisations));
    kernel.setArg(2, callBuffer);
    uint conv_p = p * std::numeric_limits<uint>::max();
    kernel.setArg(3, walksPerKernel);
    kernel.setArg(4, conv_p);
    kernel.setArg(5, optionPrice);
    kernel.setArg(6, steps);
    srand(seed);
    kernel.setArg(7, (uint)rand());
    kernel.setArg(8, (uint)rand());

    // Iteration Space
    cl::NDRange offset(0);
    cl::NDRange globalSize;
    if (antithetic) {
      globalSize = cl::NDRange(iterations / (2 * walksPerKernel));
    } else {
      globalSize = cl::NDRange(iterations);
    }

    cl::NDRange localSize = cl::NullRange;
    queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
    queue.enqueueBarrier();

    mean = calculateBufferMean(queue, callBuffer, iterations / walksPerKernel);
    callPrice = mean * exp(-finalTime * interestRate);

    timer->stop();

    calculateStatistics(queue, callBuffer);

  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
  return true;
}

void BinomialCRR_Memory::printHeaders() {
  OptionPricer::printHeaders();
  std::cout << ",Walks Per Kernel";
}

void BinomialCRR_Memory::printCSV() {
  OptionPricer::printCSV();
  std::cout << "," << walksPerKernel;
}
