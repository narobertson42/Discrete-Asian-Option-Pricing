#include "pricerRegistrar.hpp"

OptionPricerFactory::map_type *OptionPricerFactory::map =
    new OptionPricerFactory::map_type();
