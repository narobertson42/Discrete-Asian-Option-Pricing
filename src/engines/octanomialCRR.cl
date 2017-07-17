
__kernel void discreteMonteCarloKernel(
    __global float *callSums, float const optionPrice, int const steps,
    uint const seedA, uint const seedB, float const stockPrice,
    __global float *globalShifts, __local float *shifts,
    __global uint *globalProbabilities, __local uint *probabilities) {

  async_work_group_copy(shifts, globalShifts, 8, 0);
  async_work_group_copy(probabilities, globalProbabilities, 8, 0);

  float cumulativeWalk = 0.0;
  float currentValue = stockPrice;

  uint n = get_global_id(0);
  uint2 seed2 = (uint2)(seedB ^ seedA * n, seedA ^ seedB * (n + 1));
  uint random = MWC64X(&seed2);

  for (int j = 0; j < steps; j++) {
    cumulativeWalk += currentValue;
    random = MWC64X(&seed2);
    if (random <= probabilities[3]) {
      if (random <= probabilities[1]) {
        if (random <= probabilities[0]) {
          currentValue = currentValue * shifts[0];
        } else {
          currentValue = currentValue * shifts[1];
        }
      } else {
        if (random <= probabilities[2]) {
          currentValue = currentValue * shifts[2];
        } else {
          currentValue = currentValue * shifts[3];
        }
      }
    } else {
      if (random <= probabilities[5]) {
        if (random <= probabilities[4]) {
          currentValue = currentValue * shifts[4];
        } else {
          currentValue = currentValue * shifts[5];
        }
      } else {
        if (random <= probabilities[6]) {
          currentValue = currentValue * shifts[6];
        } else {
          currentValue = currentValue * shifts[7];
        }
      }
    }
    cumulativeWalk += currentValue;
  }

  callSums[n] = generatePayoff(optionPrice, cumulativeWalk, 2 * (steps));
}

__kernel void discreteMonteCarloKernelAntithetic(
    __global float *callSums, float const optionPrice, int const steps,
    uint const seedA, uint const seedB, float const stockPrice,
    __global float *globalShifts, __local float *shifts,
    __global uint *globalProbabilities, __local uint *probabilities) {

  async_work_group_copy(shifts, globalShifts, 8, 0);
  async_work_group_copy(probabilities, globalProbabilities, 8, 0);

  float cumulativeWalk = 0;
  float cumulativeWalk2 = 0;
  float currentValue = stockPrice;
  float currentValue2 = stockPrice;

  uint n = get_global_id(0);
  uint iterations = get_global_size(0);
  uint2 seed2 = (uint2)(seedB ^ seedA * n, seedA ^ seedB * (n + 1));
  uint random = MWC64X(&seed2);

  for (int j = 0; j < steps; j++) {
    cumulativeWalk += currentValue;
    random = MWC64X(&seed2);
    if (random <= probabilities[3]) {
      if (random <= probabilities[1]) {
        if (random <= probabilities[0]) {
          currentValue = currentValue * shifts[0];
        } else {
          currentValue = currentValue * shifts[1];
        }
      } else {
        if (random <= probabilities[2]) {
          currentValue = currentValue * shifts[2];
        } else {
          currentValue = currentValue * shifts[3];
        }
      }
    } else {
      if (random <= probabilities[5]) {
        if (random <= probabilities[4]) {
          currentValue = currentValue * shifts[4];
        } else {
          currentValue = currentValue * shifts[5];
        }
      } else {
        if (random <= probabilities[6]) {
          currentValue = currentValue * shifts[6];
        } else {
          currentValue = currentValue * shifts[7];
        }
      }
    }
    cumulativeWalk += currentValue;

    cumulativeWalk2 += currentValue2;
    random = 0xffffffff - random;
    if (random <= probabilities[3]) {
      if (random <= probabilities[1]) {
        if (random <= probabilities[0]) {
          currentValue2 = currentValue2 * shifts[0];
        } else {
          currentValue2 = currentValue2 * shifts[1];
        }
      } else {
        if (random <= probabilities[2]) {
          currentValue2 = currentValue2 * shifts[2];
        } else {
          currentValue2 = currentValue2 * shifts[3];
        }
      }
    } else {
      if (random <= probabilities[5]) {
        if (random <= probabilities[4]) {
          currentValue2 = currentValue2 * shifts[4];
        } else {
          currentValue2 = currentValue2 * shifts[5];
        }
      } else {
        if (random <= probabilities[6]) {
          currentValue2 = currentValue2 * shifts[6];
        } else {
          currentValue2 = currentValue2 * shifts[7];
        }
      }
    }
    cumulativeWalk2 += currentValue2;
  }

  callSums[n] = generatePayoff(optionPrice, cumulativeWalk, 2 * (steps));
  callSums[iterations + n] =
      generatePayoff(optionPrice, cumulativeWalk2, 2 * steps);
}
