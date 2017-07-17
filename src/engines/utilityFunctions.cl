#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

static uint MWC64X(uint2 *state) {
  enum { A = 4294883355U };
  uint x = (*state).x, c = (*state).y; // Unpack the state
  uint res = x ^ c;                    // Calculate the result
  uint hi = mul_hi(x, A);              // Step the RNG
  x = x * A + c;
  c = hi + (x < c);
  *state = (uint2)(x, c); // Pack the state back up
  return res;             // Return the next result
}

uint xorshift128(uint4 *state)
{
	uint t = (*state).w;
	t ^= t << 11;
	t ^= t >> 8;
	(*state).w = (*state).z;
  (*state).z = (*state).y;
  (*state).y = (*state).x;
	t ^= (*state).x;
	t ^= (*state).x >> 19;
	(*state).x = t;
	return t;
}

uint xorshift32(uint *state) {
  uint x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

// Double equivalent not supported on this device
float2 generateNormals(uint2 *seed, uint2 *seed2) {
  float u1 = 0.5;
  float u2 = 0.5;
  float a = 0.0;
  float b = 0.0;
  float normal1 = 0.0;
  float normal2 = 0.0;
  uint random1 = MWC64X(seed);
  if (random1 == 0) {
    random1 = MWC64X(seed);
  }
  uint random2 = MWC64X(seed2);

  u1 = (float)random1 / (float)0xffffffff;
  u2 = (float)random2 / (float)0xffffffff;

  a = native_sqrt(-2 * native_log(u1));
  b = 2 * M_PI * u2;

  normal1 = a * native_sin(b);
  normal2 = a * native_cos(b);

  return (float2)(normal1, normal2);
}

float generatePayoff(float const optionPrice, float const cumulativeWalk,
                     uint const steps) {
  float averageValue = cumulativeWalk / steps;
  float payOff = averageValue - optionPrice;
  if (payOff < 0.0) {
    return 0.0;
  } else {
    return payOff;
  }
}

float generateGeometricPayoff(float const optionPrice,
                              float const cumulativeWalk, uint const steps) {
  float averageValue = native_exp(cumulativeWalk / steps);
  float payOff = averageValue - optionPrice;
  if (payOff < 0.0) {
    return 0.0;
  } else {
    return payOff;
  }
}

__kernel void averageKernel(__global float *input, __global float *output,
                            uint const divisionSize, uint const inputSize) {
  uint n = get_global_id(0);
  float outputInternal = 0;
  if ((n + 1) * divisionSize <= inputSize) {
    for (uint i = 0; i < divisionSize; i++) {
      outputInternal += input[n * divisionSize + i];
    }
  } else {
    for (uint i = 0; i < (n + 1) * divisionSize - inputSize; i++) {
      outputInternal += input[n * divisionSize + i];
    }
  }
  output[n] = outputInternal;
}

__kernel void varianceKernel(__global float *input, __global float *output,
                             uint const divisionSize, uint const inputSize,
                             float const mean) {
  uint n = get_global_id(0);
  float outputInternal = 0;
  if ((n + 1) * divisionSize <= inputSize) {
    for (uint i = 0; i < divisionSize; i++) {
      outputInternal += (input[n * divisionSize + i] - mean) *
                        (input[n * divisionSize + i] - mean);
    }
  } else {
    for (uint i = 0; i < (n + 1) * divisionSize - inputSize; i++) {
      outputInternal += (input[n * divisionSize + i] - mean) *
                        (input[n * divisionSize + i] - mean);
    }
  }
  output[n] = outputInternal;
}

__kernel void controlVariateCombine(__global float *input1,
                                    __global float *input2,
                                    __global float *output,
                                    float const trueValue, float const c) {
  uint n = get_global_id(0);
  output[n] = input1[n] + c * (input2[n] - trueValue);
}

__kernel void covarianceKernel(__global float *input1, __global float *input2,
                               __global float *output, uint const divisionSize,
                               uint const inputSize, float const mean1,
                               float const mean2) {
  uint n = get_global_id(0);
  float outputInternal = 0;
  if ((n + 1) * divisionSize <= inputSize) {
    for (uint i = 0; i < divisionSize; i++) {
      outputInternal += (input1[n * divisionSize + i] - mean1) *
                        (input2[n * divisionSize + i] - mean2);
    }
  } else {
    for (uint i = 0; i < (n + 1) * divisionSize - inputSize; i++) {
      outputInternal += (input1[n * divisionSize + i] - mean1) *
                        (input2[n * divisionSize + i] - mean2);
    }
  }
  output[n] = outputInternal;
}
