// fluid_pod: AMReX plotfile -> HDF5 conversion and proper orthogonal decomposition.
#include "Commands.h"

#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    return pod::run_cli(args);
}
