#pragma once

#include "FluidGrid.h"
#include <string_view>

using namespace std;

// Deterministic visual initializers, not validated periodic CFD solutions.
enum class FlowPattern { Still, Gradient, Center, Inward, Vortex };
FlowPattern parse_pattern(string_view name); // Unknown names throw invalid_argument.
string_view pattern_name(FlowPattern pattern); // Invalid enums throw invalid_argument.

// Reset every cell; pressure/forces zero, uniform density (kg/m^3) and viscosity
// (m^2/s). Reject invalid pattern, density <= 0 or viscosity < 0/nonfinite.
void initialize_flow(FluidGrid& grid, FlowPattern pattern,
                     double density = 1.0, double viscosity = 0.01);
