__kernel void discreteMonteCarloKernel(__global float *callSums,
                                       __global float *geometricCallSums,
                                       float const optionPrice, int const steps,
                                       uint const seedA, uint const seedB, float const stockPrice,
                                       float const interestRate,
                                       float const sigma, float const dt) {

  float cumulativeWalk1 = 0;
  float cumulativeWalk2 = 0;
  float priorGeometricValue1 = native_log(stockPrice);
  float priorGeometricValue2 = native_log(stockPrice);
  float geometricCumulativeWalk1 = 0.0;
  float geometricCumulativeWalk2 = 0.0;
  float currentValue1 = stockPrice;
  float currentValue2 = stockPrice;

  uint n = get_global_id(0);
  uint iterations = get_global_size(0);
  uint2 seed2 = (uint2) (seedB ^seedA*n,seedA ^ seedB*(n+1));
  uint random1 = MWC64X(&seed2);
  uint2 seed3 = (uint2)(random1 ^seedA, random1 ^ seedB*n);
  uint random2 = MWC64X(&seed3);

  float alpha = (interestRate - 0.5 * sigma * sigma) * dt;
  float beta = sigma * sqrt(dt);

  for (int j = 0; j < steps; j++) {
    cumulativeWalk1 += currentValue1;
    cumulativeWalk2 += currentValue2;
    geometricCumulativeWalk1 += priorGeometricValue1;
    geometricCumulativeWalk2 += priorGeometricValue2;

    float2 normals;
    normals = generateNormals(&seed2, &seed3);
    float normal1 = normals.x;
    float normal2 = normals.y;

    float exp1 = native_exp(alpha + beta * normal1);
    currentValue1 = currentValue1 * exp1;
    cumulativeWalk1 += currentValue1;
    priorGeometricValue1 = native_log(currentValue1);
    geometricCumulativeWalk1 += priorGeometricValue1;

    float exp2 = native_exp(alpha + beta * normal2);
    currentValue2 = currentValue2 * exp2;
    cumulativeWalk2 += currentValue2;
    priorGeometricValue2 = native_log(currentValue2);
    geometricCumulativeWalk2 += priorGeometricValue2;
  }

  callSums[2 * n] = generatePayoff(optionPrice, cumulativeWalk1, 2 * steps);
  geometricCallSums[2 * n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk1, 2*steps);
  callSums[2 * n + 1] = generatePayoff(optionPrice, cumulativeWalk2, 2 * steps);
  geometricCallSums[2 * n + 1] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk2, 2*steps);
}

__kernel void discreteMonteCarloKernelAntithetic(
    __global float *callSums, __global float *geometricCallSums,
    float const optionPrice, int const steps, uint const seedA, uint const seedB,
    float const stockPrice, float const interestRate, float const sigma,
    float const dt) {

  float cumulativeWalk1 = 0;
  float cumulativeWalk2 = 0;
  float cumulativeWalk3 = 0;
  float cumulativeWalk4 = 0;
  float currentValue1 = stockPrice;
  float currentValue2 = stockPrice;
  float currentValue3 = stockPrice;
  float currentValue4 = stockPrice;

  float geometricCumulativeWalk1 = 0.0;
  float geometricCumulativeWalk2 = 0.0;
  float geometricCumulativeWalk3 = 0.0;
  float geometricCumulativeWalk4 = 0.0;

  float priorGeometricValue1 = native_log(stockPrice);
  float priorGeometricValue2 = native_log(stockPrice);
  float priorGeometricValue3 = native_log(stockPrice);
  float priorGeometricValue4 = native_log(stockPrice);

  uint n = get_global_id(0);
  uint iterations = get_global_size(0);
  uint2 seed2 = (uint2) (seedB ^seedA*n,seedA ^ seedB*(n+1));
  uint random1 = MWC64X(&seed2);
  uint2 seed3 = (uint2)(random1 ^seedA, random1 ^ seedB*n);
  uint random2 = MWC64X(&seed3);

  float alpha = (interestRate - 0.5 * sigma * sigma) * dt;
  float beta = sigma * sqrt(dt);

  for (int j = 0; j < steps; j++) {
    cumulativeWalk1 += currentValue1;
    cumulativeWalk2 += currentValue2;
    cumulativeWalk3 += currentValue3;
    cumulativeWalk4 += currentValue4;

    geometricCumulativeWalk1 += priorGeometricValue1;
    geometricCumulativeWalk2 += priorGeometricValue2;
    geometricCumulativeWalk3 += priorGeometricValue3;
    geometricCumulativeWalk4 += priorGeometricValue4;

    float2 normals;
    normals = generateNormals(&seed2, &seed3);
    float normal1 = normals.x;
    float normal2 = normals.y;

    float exp1 = native_exp(alpha + beta * normal1);
    currentValue1 = currentValue1 * exp1;

    float exp2 = native_exp(alpha + beta * normal2);
    currentValue2 = currentValue2 * exp2;

    float exp3 = native_exp(alpha + beta * -normal1);
    currentValue3 = currentValue3 * exp3;

    float exp4 = native_exp(alpha + beta * -normal2);
    currentValue4 = currentValue4 * exp4;

    cumulativeWalk1 += currentValue1;
    priorGeometricValue1 = native_log(currentValue1);
    geometricCumulativeWalk1 += priorGeometricValue1;

    cumulativeWalk2 += currentValue2;
    priorGeometricValue2 = native_log(currentValue2);
    geometricCumulativeWalk2 += priorGeometricValue2;

    cumulativeWalk3 += currentValue3;
    priorGeometricValue3 = native_log(currentValue3);
    geometricCumulativeWalk3 += priorGeometricValue3;

    cumulativeWalk4 += currentValue4;
    priorGeometricValue4 = native_log(currentValue4);
    geometricCumulativeWalk4 += priorGeometricValue4;

  }

  callSums[2 * n] = generatePayoff(optionPrice, cumulativeWalk1, 2 * steps);
  callSums[2 * n + 1] = generatePayoff(optionPrice, cumulativeWalk2, 2 * steps);
  callSums[2 * iterations + 2 * n] =
      generatePayoff(optionPrice, cumulativeWalk3, 2 * steps);
  callSums[2 * iterations + 2 * n + 1] =
      generatePayoff(optionPrice, cumulativeWalk4, 2 * steps);

  geometricCallSums[2 * n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk1, 2 * steps);
  geometricCallSums[2 * n + 1] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk2,2 * steps);
  geometricCallSums[2 * iterations + 2 * n] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk3, 2 * steps);
  geometricCallSums[2 * iterations + 2 * n + 1] =
      generateGeometricPayoff(optionPrice, geometricCumulativeWalk4, 2 * steps);
}
