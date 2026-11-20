// Read side of the snapshot HDF5 layout written by `fluid_pod convert` (format "eigenflow-snapshots").
#pragma once
#include "H5Util.h"
#include "VtkHdf.h"

#include <string>
#include <vector>

namespace pod {

inline const char* kSnapshotFormat = "eigenflow-snapshots";
inline const char* kPodFormat = "eigenflow-pod";

struct SeriesMeta {
    std::string name;
    std::vector<std::string> vars;
    std::vector<double> x, y, time;
    std::vector<long long> step;
    double prob_lo[2]{}, dx[2]{};
    int coord_sys = 0;
    int itype[2]{};
    std::string centering = "cell";
    int nx() const { return (int)x.size(); }
    int ny() const { return (int)y.size(); }
    int nt() const { return (int)time.size(); }
};

std::vector<std::string> list_series(hid_t file);  // groups that contain a "fields" group
SeriesMeta read_series(hid_t file, const std::string& name);
// Resolve -s: exact name, glob matching exactly one series, or the only series in the file.
std::string pick_series(hid_t file, const std::string& requested, const std::string& path);

// Geometry/centering attributes shared by snapshot series and POD files.
void write_geometry_attrs(hid_t obj, const double prob_lo[2], const double dx[2], int coord_sys, const int itype[2],
                          const std::vector<double>& x, const std::vector<double>& y);
VtkGrid vtk_grid(const std::vector<double>& x, const std::vector<double>& y, const double dx[2],
                   const int itype[2]);

}  // namespace pod
