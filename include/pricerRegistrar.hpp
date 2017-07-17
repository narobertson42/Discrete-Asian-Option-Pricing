// Based on http://stackoverflow.com/a/582456

#ifndef PRICER_REGISTRAR_HEADER
#define PRICER_REGISTRAR_HEADER

#include "optionPricer.hpp"

#include <map>
#include <string>

struct OptionPricerFactory {
  typedef std::map<std::string, OptionPricer *(*)()> map_type;
  static map_type *map;

  static OptionPricer *createInstance(std::string const &s) {

    map_type::iterator it = getMap()->find(s);
    if (it == getMap()->end()) {
      std::cout << "Simulation identifier not recognised" << std::endl;
      return 0;
    }
    return it->second();
  }

protected:
  static map_type *getMap() {
    if (!map) {
      map = new map_type;
    }
    return map;
  }
};

template <typename T> struct DerivedRegister : OptionPricerFactory {
  DerivedRegister(std::string const &s) {
    getMap()->insert(
        std::pair<std::string, OptionPricer *(*)()>(s, &createT<T>));
  }
};

#endif // PRICER_REGISTRAR_HEADER
