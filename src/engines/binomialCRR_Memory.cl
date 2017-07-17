
__kernel void discreteMonteCarloKernel(__global float *discreteValues,
                                       __local float *localDiscreteValues,
                                       __global float *callSums,
                                       int const kernelRepetitions,
                                       uint const p, float const optionPrice,
                                       int const steps, uint const seedA, uint const seedB) {

  async_work_group_copy(localDiscreteValues, discreteValues, (2 * steps) + 1,
                        0);

  float cumulativeWalk = 0.0;
  int currentPoint = steps;

  uint n = get_global_id(0);
  uint2 seed2 = (uint2) (seedB ^seedA*n,seedA ^ seedB*(n+1));
  uint random = MWC64X(&seed2); // 32 divide

  float cumulativeCallValue = 0.0;

  for (int i = 0; i < kernelRepetitions; i++) {
    random = MWC64X(&seed2);
    currentPoint = steps;
    cumulativeWalk = 0.0;
    for (int j = 0; j < steps; j++) {
      cumulativeWalk += localDiscreteValues[currentPoint];
      random = MWC64X(&seed2);
      if (random < p) {
        currentPoint++;
      } else {
        currentPoint--;
      }
      cumulativeWalk += localDiscreteValues[currentPoint];
    }
    cumulativeCallValue +=
        generatePayoff(optionPrice, cumulativeWalk, 2 * steps);
  }

  callSums[n] = cumulativeCallValue / (float)kernelRepetitions;
}


__kernel void discreteMonteCarloKernelAntithetic(__global float *discreteValues,
                                       __local float *localDiscreteValues,
                                       __global float *callSums,
                                       int const kernelRepetitions,
                                       uint const p, float const optionPrice,
                                       int const steps, uint const seedA, uint const seedB) {

  async_work_group_copy(localDiscreteValues, discreteValues, (2 * steps) + 1,
                        0);

  float cumulativeWalk = 0.0;
  int currentPoint = steps;
  float cumulativeWalk2 = 0.0;
  int currentPoint2 = steps;

  uint n = get_global_id(0);
  uint iterations = get_global_size(0);
  uint2 seed2 = (uint2) (seedB ^seedA*n,seedA ^ seedB*(n+1));
  uint random = MWC64X(&seed2); // 32 divide

  float cumulativeCallValue = 0.0;
  float cumulativeCallValue2 = 0.0;

  for (int i = 0; i < kernelRepetitions; i++) {
    random = MWC64X(&seed2);
    currentPoint = steps;
    cumulativeWalk = 0.0;
    currentPoint2 = steps;
    cumulativeWalk2 = 0.0;
    for (int j = 0; j < steps; j++) {
      cumulativeWalk += localDiscreteValues[currentPoint];
      cumulativeWalk2 += localDiscreteValues[currentPoint2];
      random = MWC64X(&seed2);
      if (random < p) {
        currentPoint++;
      } else {
        currentPoint--;
      }
      if (0xffffffff - random < p) {
        currentPoint2++;
      } else {
        currentPoint2--;
      }
      cumulativeWalk += localDiscreteValues[currentPoint];
      cumulativeWalk2 += localDiscreteValues[currentPoint2];
    }
    cumulativeCallValue +=
        generatePayoff(optionPrice, cumulativeWalk, 2 * steps);
        cumulativeCallValue2 +=
            generatePayoff(optionPrice, cumulativeWalk2, 2 * steps);
  }

  callSums[n] = cumulativeCallValue / (float)kernelRepetitions;
  callSums[n+iterations] = cumulativeCallValue2 / (float)kernelRepetitions;
}
