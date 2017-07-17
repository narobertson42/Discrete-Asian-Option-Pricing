
__kernel void discreteMonteCarloKernel(__global float *callSums, uint const p,
                                       float const optionPrice, int const steps,
                                       uint const seedA, uint const seedB, float const stockPrice,
                                       float const u, float const d) {

  float cumulativeWalk = 0.0;
  float currentValue = stockPrice;

  uint n = get_global_id(0);
  uint2 seed2 = (uint2) (seedB ^seedA*n,seedA ^ seedB*(n+1));
  uint random = MWC64X(&seed2);

  for (int j = 0; j < steps; j++) {
    cumulativeWalk += currentValue;
    random = MWC64X(&seed2);
    currentValue = random < p ? currentValue*u : currentValue*d;
    cumulativeWalk += currentValue;
  }

  callSums[n] = generatePayoff(optionPrice, cumulativeWalk, 2 * (steps));
}

__kernel void
discreteMonteCarloKernelAntithetic(__global float *callSums, uint const p,
                                   float const optionPrice, int const steps,
                                   uint const seedA, uint const seedB, float const stockPrice,
                                   float const u, float const d) {

  float cumulativeWalk = 0.0;
  float cumulativeWalk2 = 0.0;
  float currentValue = stockPrice;
  float currentValue2 = stockPrice;

  uint n = get_global_id(0);
  uint iterations = get_global_size(0);
  uint2 seed2 = (uint2) (seedB ^seedA*n,seedA ^ seedB*(n+1));
  uint random = MWC64X(&seed2);
  uint pinv = 0xffffffff - p;

  for (int j = 0; j < steps; j++) {
    random = MWC64X(&seed2);
    cumulativeWalk += currentValue;
    cumulativeWalk2 += currentValue2;
    currentValue = random < p ? currentValue*u : currentValue*d;
    currentValue2 = pinv < random ? currentValue2*u : currentValue2*d;
    cumulativeWalk += currentValue;
    cumulativeWalk2 += currentValue2;
  }

  callSums[n] = generatePayoff(optionPrice, cumulativeWalk, 2 * steps);
  callSums[iterations + n] =
      generatePayoff(optionPrice, cumulativeWalk2, 2 * steps);
}
