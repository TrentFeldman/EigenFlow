// Dimensionless preset shapes sampled at physical cell centers.
#include "FlowPresets.h"
#include <cmath>
#include <stdexcept>

using namespace std;

FlowPattern parse_pattern(string_view name) {
    for (auto pattern : {FlowPattern::Still, FlowPattern::Gradient, FlowPattern::Center,
                         FlowPattern::Inward, FlowPattern::Vortex})
        if (name == pattern_name(pattern)) return pattern;
    throw invalid_argument("Pattern must be still, gradient, center, inward, or vortex.");
}

string_view pattern_name(FlowPattern pattern) {
    switch (pattern) {
    case FlowPattern::Still: return "still";
    case FlowPattern::Gradient: return "gradient";
    case FlowPattern::Center: return "center";
    case FlowPattern::Inward: return "inward";
    case FlowPattern::Vortex: return "vortex";
    }
    throw invalid_argument("Unknown flow pattern.");
}

void initialize_flow(FluidGrid& grid, FlowPattern pattern, double density, double viscosity) {
    (void)pattern_name(pattern);
    if (!isfinite(density) || density <= 0 || !isfinite(viscosity) || viscosity < 0)
        throw invalid_argument("Density must be positive and viscosity nonnegative, both finite.");
    for (size_t y = 0; y < grid.height(); ++y) {
        const double vertical = 2.0 * (static_cast<double>(y) + 0.5) /
                                static_cast<double>(grid.height()) - 1.0;
        for (size_t x = 0; x < grid.width(); ++x) {
            const double horizontal = 2.0 * (static_cast<double>(x) + 0.5) /
                                      static_cast<double>(grid.width()) - 1.0;
            Cell cell;
            cell.density = density;
            cell.kinematic_viscosity = viscosity;
            switch (pattern) {
            case FlowPattern::Still: break;
            case FlowPattern::Gradient: cell.velocity_x = (vertical + 1.0) / 2.0; break;
            case FlowPattern::Center:
                cell.velocity_x = horizontal; cell.velocity_y = vertical; break;
            case FlowPattern::Inward:
                cell.velocity_x = -horizontal; cell.velocity_y = -vertical; break;
            case FlowPattern::Vortex:
                cell.velocity_x = -vertical; cell.velocity_y = horizontal; break;
            }
            grid.at(x, y) = cell;
        }
    }
}
