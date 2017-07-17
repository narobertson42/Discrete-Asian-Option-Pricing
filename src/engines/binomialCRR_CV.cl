__kernel void discreteMonteCarloKernel(__global float *callSums,
                                       __global float *geometricCallSums,
                                       float const optionPrice, int const steps,
                                       uint const seedA, uint const seedB,
                                       float const stockPrice, float const u,
                                       float const d, uint const p) {
  float cumulativeWalk = 0;
  float priorGeometricValue = native_log(stockPrice);
  float geometricCumulativeWalk = 0;
  float currentValue = stockPrice;

  uint n = get_global_id(0);
  uint iterations = get_global_size(0);

  uint2 seed2 = (uint2)(seedB ^ seedA * n, seedA ^ seedB * (n + 1));
  uint random = MWC64X(&seed2);

  for (int j = 0; j < steps; j++) {
    random = MWC64X(&seed2);
    cumulativeWalk += currentValue;
    geometricCumulativeWalk += priorGeometricValue;
    if (random < p) {
      currentValue = currentValue * u;
    } else {
      currentValue = currentValue * d;
    }
    priorGeometricValue = native_log(currentValue);
    cumulativeWalk += currentValue;
    geometricCumulativeWalk += priorGeometricValue;
  }

  callSums[n] = generatePayoff(optionPrice, cumulativeWalk, 2 * steps);
  geometricCallSums[n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk, 2 * steps);
}

__kernel void discreteMonteCarloKernelAntithetic(
    __global float *callSums, __global float *geometricCallSums,
    float const optionPrice, int const steps, uint const seedA,
    uint const seedB, float const stockPrice, float const u, float const d,
    uint const p) {
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
  uint pinv = 0xffffffff - p;

  uint2 seed2 = (uint2)(seedB ^ seedA * n, seedA ^ seedB * (n + 1));
  uint random = MWC64X(&seed2);

  for (int j = 0; j < steps; j++) {
    random = MWC64X(&seed2);
    cumulativeWalk += currentValue;
    cumulativeWalk2 += currentValue2;
    geometricCumulativeWalk += priorGeometricValue;
    geometricCumulativeWalk2 += priorGeometricValue2;
    if (random <p) {
      currentValue = currentValue * u;
    } else {
      currentValue = currentValue * d;
    }
    if (pinv <random) {
      currentValue2 = currentValue2 * u;
    } else {
      currentValue2 = currentValue2 * d;
    }
    priorGeometricValue = native_log(currentValue);
    priorGeometricValue2 = native_log(currentValue2);
    cumulativeWalk += currentValue;
    cumulativeWalk2 += currentValue2;
    geometricCumulativeWalk += priorGeometricValue;
    geometricCumulativeWalk2 += priorGeometricValue2;

  }
  callSums[n] = generatePayoff(optionPrice, cumulativeWalk, 2 * steps);
  geometricCallSums[n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk, 2 * steps);
  callSums[iterations + n] =
      generatePayoff(optionPrice, cumulativeWalk2, 2 * steps);
  geometricCallSums[iterations + n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk2, 2 * steps);
}
