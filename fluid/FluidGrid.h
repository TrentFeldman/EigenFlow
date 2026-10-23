#pragma once

#include <cstddef>
#include <vector>

using namespace std;

// All fields are sampled at cell centers, in SI units. No thermal state yet.
struct Cell {
    double velocity_x = 0.0;          // m/s
    double velocity_y = 0.0;
    double pressure = 0.0;            // Gauge pressure, Pa.
    double density = 1.0;             // kg/m^3; uniform for this fluid.
    double kinematic_viscosity = 0.01; // m^2/s; uniform for this fluid.
    double force_x = 0.0;             // Force per volume, N/m^3.
    double force_y = 0.0;
};

// Physical rectangle in metres; both axes use periodic neighbor indexing.
struct Domain {
    double length_x = 2.0;
    double length_y = 2.0;
    double origin_x = -1.0;
    double origin_y = -1.0;

    // Reject nonfinite, nonpositive or unrepresentable extents.
    void validate() const;
};

// Owns cells[y][x]. Storage has no solver or output dependencies.
class FluidGrid {
public:
    // Positive dimensions; blank velocity/pressure/force and default material.
    // Invalid geometry throws invalid_argument; allocation failures propagate.
    explicit FluidGrid(size_t width = 64, size_t height = 64, Domain domain = {});
    size_t width() const noexcept;
    size_t height() const noexcept;
    const Domain& domain() const noexcept;
    double dx() const noexcept;
    double dy() const noexcept;

    // Cell-center positions in metres; invalid indices throw out_of_range.
    double x_center(size_t x) const;
    double y_center(size_t y) const;
    Cell& at(size_t x, size_t y);
    const Cell& at(size_t x, size_t y) const;

    // Signed indices wrap across either edge, including multiple periods.
    Cell& periodic_at(ptrdiff_t x, ptrdiff_t y);
    const Cell& periodic_at(ptrdiff_t x, ptrdiff_t y) const;
    bool same_geometry(const FluidGrid& other) const noexcept;

private:
    size_t width_;
    size_t height_;
    Domain domain_;
    vector<vector<Cell>> cells_;
};
