#include "pricerRegistrar.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  try {
    if (argc != 11) {
      std::cout << "Input Arguments Required:\n"
                   "1. Stock Price\n"
                   "2. Strike Price\n"
                   "3. Interest\n"
                   "4. Final Time\n"
                   "5. Volatility\n"
                   "6. Steps\n"
                   "7. Iterations\n"
                   "8. Random Seed (BOOL)\n"
                   "9. Antithetic (BOOL)\n"
                   "10. Engine Name"
                << std::endl;
    } else {
      // Load in Variables
      float S = atof(argv[1]);         // Stock Price
      float K = atof(argv[2]);         // Option Price
      float r = atof(argv[3]);         // Interest Rate
      float finalTime = atof(argv[4]); // Final Time
      float sigma = atof(argv[5]);     // Volatility
      int M = atoi(argv[6]);           // Steps
      int N = atoi(argv[7]);           // Iterations
      int randomSeed = atoi(argv[8]);  // Truly Random or Not
      int antithetic = atoi(argv[9]);  // Antithetic
      std::string pricerName = argv[10];

      OptionPricer *optionPricer =
          OptionPricerFactory::createInstance(pricerName);

      if (optionPricer == 0) {
        throw 42;
      }

      if (randomSeed == 1) {
        srand(time(NULL));
      } else {
        srand(42);
      }
      optionPricer->UpdateParameters(S, K, r, finalTime, sigma, M, rand(),
                                     antithetic);
      optionPricer->execute(N);
      optionPricer->terminalFriendlyPrint();
      // optionPricer->printHeaders();
      // optionPricer->printCSV();
      std::cout << "\n";
    }
    return 0;
  } catch (int i) {
    if (i == 42) {
      std::cout << "\nAVAILABLE ENGINES" << std::endl;
      OptionPricerFactory::map_type *map = OptionPricerFactory::map;
      for (OptionPricerFactory::map_type::iterator it = map->begin();
           it != map->end(); ++it) {
        std::cout << string(it->first) << std::endl;
      }
    } else {
      std::cout << "Error:" << i << std::endl;
    }
  } catch (...) {
    std::cout << "Unknown Error" << std::endl;
  }
}
