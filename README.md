# Discrete Asian Option Pricing for GPUs
This repository is a **simplified** version of the code written for the included Final Year Project [Report, "Asian Option Pricing Techniques for GPUs"](../blob/master/Report.pdf). It has been simplified by stripping extraneous option pricing engines, testing scripts, and anything else deemed superfluous to understanding the underlying algorithms.

## Report Abstract

Option pricing is often a computationally intensive process, particularly when the option requires the use of Monte Carlo based pricing methods. Last year, Pieter Fabry developed a new method based on Monte Carlo pricing over Cox-Ross-Rubinstein binomial trees, using random asset paths based on flipping a biased coin, rather than generating a log-normal distributed sample. He discovered that the parallelisation benefits of the discrete model outweighed the expected losses in accuracy hardware for FPGAs.

This project concerns the analysis and implementation of discrete space pricing schemes for GPUs. The project considers whether the parallelisation advantages posited for FPGAs hold for GPUs. The basis of this discrete-space investigation considers Jarrow-Rudd binomial trees, multinomial trees, antithetic sampling, and control variates.

It is found that normally  multinomial trees can be constructed with the same convergence properties as a continuous Gaussian walk. On a GPU, the continuous model can be replaced with a sampled multinomial tree with 64 discretisations - this resulted in a 3.14× increase in throughput compared to the continuous equivalent. Antithetic sampling improved the throughput increase to 3.65×. The addition of a geometric control variate as well as the antithetic variate resulted in an 18.82× speed up for a 99% confidence interval of size 2 × 10−3.

## Framework Structure

![UML-esque Diagram](../blob/master/TestSuiteArchitecture.png?raw=true)

## Getting Started

You will need to have OpenCL and C++ Boost libraries setup in order to build this project.

```
make all
```

To check that the build was successful, run the simplisticTest executable to check everything runs. N.B. There is a 2% chance there will be at least one false positive FAIL message.

```
./bin/simplisticTest
```

Selecting the execution device and platform is done through use of terminal environment variables

```
export MONTE_CARLO_PLATFORM=0
export MONTE_CARLO_DEVICE=1
```

Running a specific engine is completed using the specificSimulation executable. It takes command line arguments to determine the execution properties. The arguments required are:
1. Stock Price
2. Strike Price
3. Interest Rate
4. Final Time (Years)
5. Volatility
6. Time Step Discretisations
7. Monte Carlo Iterations
8. Random Seed (BOOL)
9. Antithetic (BOOL)
10. Engine Name (Entering an incorrect engine name will provide a list of the available engines)

```
./bin/specificSimulation 100 110 0.15 1 0.3 200 20000000 1 1 BinomialCRR_CV 2> /dev/null
```
