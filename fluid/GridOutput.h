#pragma once

#include "FluidGrid.h"
#include <cstdint>
#include <iostream>
#include <string>

using namespace std;

// Read-only exporters. Parent directories must exist. File exporters overwrite
// their explicit path; invalid data throws invalid_argument before opening,
// I/O errors throw runtime_error and may leave partial output.
void print_grid(const FluidGrid& grid, ostream& out = cout);

// Seven float64 fields shaped [height,width], with SI geometry and time metadata.
// time is seconds, step counts completed updates; time must be finite and >= 0.
void save_hdf5(const FluidGrid& grid, const string& path,
               double time = 0.0, uint64_t step = 0);

// Index-space chart: exact velocity directions, per-chart relative speed scale.
void save_svg(const FluidGrid& grid, const string& path);
