#include "pricerRegistrar.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  // Load in Variables
  float S = 100.0;       // Stock Price
  float K = 110.0;       // Option Price
  float r = 0.15;        // Interest Rate
  float finalTime = 1.0; // Final Time
  float sigma = 0.3;     // Volatility
  int M = 1000;          // Steps
  int N = 1000000;       // Iterations

  std::vector<std::string> optionPricers;
  for (map<std::string, OptionPricer *(*)()>::iterator it =
           OptionPricerFactory::map->begin();
       it != OptionPricerFactory::map->end(); ++it) {
    optionPricers.push_back(it->first);
  }

  for (uint j = 0; j < 2; j++) {
    srand(time(NULL));
    for (uint i = 0; i < optionPricers.size(); i++) {
      OptionPricer *optionPricer =
          OptionPricerFactory::createInstance(optionPricers[i]);
      optionPricer->UpdateParameters(S, K, r, finalTime, sigma, M, rand(),
                                     (bool)j);
      optionPricer->testModelCorrectness();
    }
  }

  return 0;
}
