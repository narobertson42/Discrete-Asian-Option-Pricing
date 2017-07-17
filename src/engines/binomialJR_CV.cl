__kernel void discreteMonteCarloKernel(__global float *callSums,
                                       __global float *geometricCallSums,
                                       float const optionPrice, int const steps,
                                       uint const seedA, uint const seedB,
                                       float const stockPrice,
                                       __global float *globalShifts,
                                       __local float *shifts) {

  async_work_group_copy(shifts, globalShifts, 2, 0);

  float cumulativeWalk = 0;
  float priorGeometricValue = native_log(stockPrice);
  float geometricCumulativeWalk = 0;
  float currentValue = stockPrice;

  uint n = get_global_id(0);
  uint iterations = get_global_size(0);

  uint2 seed2 = (uint2)(seedB ^ seedA * n, seedA ^ seedB * (n + 1));
  uint random = MWC64X(&seed2);

  for (int j = 0; j < steps; j = j) {
    uint random = MWC64X(&seed2);
    int stepsLeft = steps - j;
    if (stepsLeft > 32) {
      stepsLeft = 32;
    }
    for (int i = 0; i < stepsLeft; i++) {
      cumulativeWalk += currentValue;
      geometricCumulativeWalk += priorGeometricValue;
      currentValue = currentValue * shifts[random & 1];
      priorGeometricValue = native_log(currentValue);
      cumulativeWalk += currentValue;
      geometricCumulativeWalk += priorGeometricValue;
      random = random >> 1;
      j++;
    }
  }

  callSums[n] = generatePayoff(optionPrice, cumulativeWalk, 2 * steps);
  geometricCallSums[n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk, 2 * steps);
}

__kernel void discreteMonteCarloKernelAntithetic(
    __global float *callSums, __global float *geometricCallSums,
    float const optionPrice, int const steps, uint const seedA,
    uint const seedB, float const stockPrice, __global float *globalShifts,
    __local float *shifts) {

  async_work_group_copy(shifts, globalShifts, 2, 0);

  float cumulativeWalk = 0;
  float cumulativeWalk2 = 0;
  float priorGeometricValue = native_log(stockPrice);
  float priorGeometricValue2 = native_log(stockPrice);
  float geometricCumulativeWalk = 0;
  float geometricCumulativeWalk2 = 0;
  float currentValue = stockPrice;
  float currentValue2 = stockPrice;

  uint n = get_global_id(0);
  uint iterations = get_global_size(0);

  uint2 seed2 = (uint2)(seedB ^ seedA * n, seedA ^ seedB * (n + 1));
  uint random = MWC64X(&seed2);

  for (int j = 0; j < steps; j = j) {
    random = MWC64X(&seed2);
    int stepsLeft = steps - j;
    if (stepsLeft > 32) {
      stepsLeft = 32;
    }
    for (int i = 0; i < stepsLeft; i++) {
      cumulativeWalk += currentValue;
      cumulativeWalk2 += currentValue2;
      geometricCumulativeWalk += priorGeometricValue;
      geometricCumulativeWalk2 += priorGeometricValue2;
      currentValue = currentValue * shifts[random & 0b1];
      currentValue2 = currentValue2 * shifts[(0xffffffff - random & 0b1)];
      priorGeometricValue = native_log(currentValue);
      priorGeometricValue2 = native_log(currentValue2);
      cumulativeWalk += currentValue;
      cumulativeWalk2 += currentValue2;
      geometricCumulativeWalk += priorGeometricValue;
      geometricCumulativeWalk2 += priorGeometricValue2;
      random = random >> 1;
      j++;
    }
  }
  callSums[n] = generatePayoff(optionPrice, cumulativeWalk, 2 * steps);
  geometricCallSums[n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk, 2 * steps);
  callSums[iterations + n] =
      generatePayoff(optionPrice, cumulativeWalk2, 2 * steps);
  geometricCallSums[iterations + n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk2, 2 * steps);
}
