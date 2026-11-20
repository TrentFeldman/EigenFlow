// ParaView sidecar in VTK's native VTKHDF format (ImageData, temporal). Array datasets are HDF5
// external links into the main file, so the sidecar is a few KiB and never duplicates field data.
// Open the .vtkhdf file in ParaView (File > Open); keep it next to the .h5 it points to.
#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace pod {

struct VtkField {
    std::string name;             // array name shown in ParaView
    std::string dataset;          // [nsteps, ny, nx] dataset in the target file (external link), or
    std::vector<double> inline_;  // ... [ny, nx] values stored in the sidecar and shown at every step
};

struct VtkGrid {
    int nx = 0, ny = 0;
    double origin[2]{};  // lower-left corner of the cell grid (or first node when node_centered)
    double spacing[2]{};
    bool node_centered = false;
};

void write_vtkhdf(const std::filesystem::path& sidecar, const std::filesystem::path& target_h5,
                  const std::vector<double>& steps, const VtkGrid& grid, const std::vector<VtkField>& fields);

}  // namespace pod
