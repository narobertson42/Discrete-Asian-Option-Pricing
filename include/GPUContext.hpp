// Handles the initialisation of the device and context associate with each
// option pricer. It also adds the utility functions written in OpenCL to each
// file.

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif

#ifndef GPU_CONTEXT_HEADER
#define GPU_CONTEXT_HEADER

#include "CL/cl.hpp"
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

class GPUContext {
public:
  GPUContext() {}
  GPUContext(std::string filename) {
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
      throw std::runtime_error("No OpenCL platforms found.");

    std::cerr << "Found " << platforms.size() << " platforms\n";
    for (unsigned i = 0; i < platforms.size(); i++) {
      std::string vendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
      std::cerr << "  Platform " << i << " : " << vendor << "\n";
    }
    int selectedPlatform = 0;
    if (getenv("MONTE_CARLO_PLATFORM")) {
      selectedPlatform = atoi(getenv("MONTE_CARLO_PLATFORM"));
    }
    std::cerr << "Choosing platform " << selectedPlatform << "\n";
    cl::Platform platform = platforms.at(selectedPlatform);

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.size() == 0) {
      throw std::runtime_error("No opencl devices found.\n");
    }

    std::cerr << "Found " << devices.size() << " devices\n";
    for (unsigned i = 0; i < devices.size(); i++) {
      std::string name = devices[i].getInfo<CL_DEVICE_NAME>();
      std::cerr << "  Device " << i << " : " << name << "\n";
    }

    int selectedDevice = 0;
    if (getenv("MONTE_CARLO_DEVICE")) {
      selectedDevice = atoi(getenv("MONTE_CARLO_DEVICE"));
    }
    std::cerr << "Choosing device " << selectedDevice << "\n";
    device = devices.at(selectedDevice);

    context = cl::Context(devices);

    // Appends the utility functions for OpenCL to the beginning of the kernel
    // file of interest
    std::string filename2 = "src/engines/utilityFunctions.cl";
    std::ifstream utilitySrc(filename2, std::ios::in | std::ios::binary);
    if (!utilitySrc.is_open())
      throw std::runtime_error("Couldn't load OpenCL Utility Functions");

    std::ifstream src(filename, std::ios::in | std::ios::binary);
    if (!src.is_open())
      throw std::runtime_error("Couldn't load OpenCL File");

    std::string kernelSource =
        std::string((std::istreambuf_iterator<char>(
                        utilitySrc)), // Node the extra brackets.
                    std::istreambuf_iterator<char>());

    kernelSource += std::string(
        (std::istreambuf_iterator<char>(src)), // Node the extra brackets.
        std::istreambuf_iterator<char>());

    cl::Program::Sources sources;
    sources.push_back(
        std::make_pair(kernelSource.c_str(), kernelSource.size() + 1));
    program = cl::Program(context, sources);

    try {
      program.build(devices); //, "-cl-finite-math-only"
    } catch (...) {
      for (unsigned i = 0; i < devices.size(); i++) {
        std::cerr << "Log for device " << devices[i].getInfo<CL_DEVICE_NAME>()
                  << ":\n\n";
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])
                  << "\n\n";
      }
      throw;
    }
  };

  // Variables
  cl::Device device;
  cl::Context context;
  cl::Program program;
};

#endif // GPU_CONTEXT_HEADER
