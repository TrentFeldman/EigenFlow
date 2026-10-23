// Grid ownership, physical coordinates and periodic neighbor access.
#include "FluidGrid.h"
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace std;

namespace {
size_t wrapped(ptrdiff_t index, size_t count) {
    const auto size = static_cast<ptrdiff_t>(count);
    const auto remainder = index % size;
    return static_cast<size_t>(remainder < 0 ? remainder + size : remainder);
}
}

void Domain::validate() const {
    const auto valid_axis = [](double origin, double length) {
        return isfinite(origin) && isfinite(length) && length > 0 &&
               isfinite(origin + length) && origin + length > origin;
    };
    if (!valid_axis(origin_x, length_x) || !valid_axis(origin_y, length_y))
        throw invalid_argument("Domain lengths must be positive, finite and representable at the origin.");
}

FluidGrid::FluidGrid(size_t width, size_t height, Domain domain)
    : width_(width), height_(height), domain_(domain) {
    domain_.validate();
    const auto index_limit = static_cast<size_t>(numeric_limits<ptrdiff_t>::max());
    if (width == 0 || height == 0 || width > index_limit || height > index_limit ||
        width > numeric_limits<size_t>::max() / height / sizeof(Cell))
        throw invalid_argument("Grid dimensions must be positive and representable.");
    if (!(dx() > 0) || !(dy() > 0))
        throw invalid_argument("Grid spacing underflows at this resolution.");
    cells_.resize(height, vector<Cell>(width));
}

size_t FluidGrid::width() const noexcept { return width_; }
size_t FluidGrid::height() const noexcept { return height_; }
const Domain& FluidGrid::domain() const noexcept { return domain_; }
double FluidGrid::dx() const noexcept { return domain_.length_x / static_cast<double>(width_); }
double FluidGrid::dy() const noexcept { return domain_.length_y / static_cast<double>(height_); }
double FluidGrid::x_center(size_t x) const {
    if (x >= width_) throw out_of_range("Column outside grid.");
    return domain_.origin_x + (static_cast<double>(x) + 0.5) * dx();
}
double FluidGrid::y_center(size_t y) const {
    if (y >= height_) throw out_of_range("Row outside grid.");
    return domain_.origin_y + (static_cast<double>(y) + 0.5) * dy();
}
Cell& FluidGrid::at(size_t x, size_t y) { return cells_.at(y).at(x); }
const Cell& FluidGrid::at(size_t x, size_t y) const { return cells_.at(y).at(x); }
Cell& FluidGrid::periodic_at(ptrdiff_t x, ptrdiff_t y) {
    return at(wrapped(x, width_), wrapped(y, height_));
}
const Cell& FluidGrid::periodic_at(ptrdiff_t x, ptrdiff_t y) const {
    return at(wrapped(x, width_), wrapped(y, height_));
}
bool FluidGrid::same_geometry(const FluidGrid& other) const noexcept {
    return width_ == other.width_ && height_ == other.height_ &&
           domain_.length_x == other.domain_.length_x &&
           domain_.length_y == other.domain_.length_y &&
           domain_.origin_x == other.domain_.origin_x &&
           domain_.origin_y == other.domain_.origin_y;
}
