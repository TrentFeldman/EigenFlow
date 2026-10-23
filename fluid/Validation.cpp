// Diagnostics and an analytical benchmark, independent of the future solver stencil.
#include "Validation.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace std;
namespace {
constexpr double material_tolerance = 1e-12;
const double pi = acos(-1.0);

double finite_result(long double value) {
    const double result = static_cast<double>(value);
    if (!isfinite(result)) throw overflow_error("Diagnostic is outside finite double range.");
    return result;
}
void validate_state(const FluidGrid& grid) {
    if (grid.width() < 3 || grid.height() < 3)
        throw invalid_argument("Periodic centered diagnostics require at least 3 cells per axis.");
    const auto& material = grid.at(0, 0);
    for (size_t y = 0; y < grid.height(); ++y) {
        for (size_t x = 0; x < grid.width(); ++x) {
            const auto& c = grid.at(x, y);
            for (double v : {c.velocity_x, c.velocity_y, c.pressure, c.density,
                             c.kinematic_viscosity, c.force_x, c.force_y})
                if (!isfinite(v)) throw invalid_argument("Nonfinite fluid field.");
            if (c.density <= 0 || c.kinematic_viscosity < 0 ||
                abs(c.density - material.density) > material_tolerance * material.density ||
                abs(c.kinematic_viscosity - material.kinematic_viscosity) >
                    material_tolerance * max(c.kinematic_viscosity, material.kinematic_viscosity))
                throw invalid_argument("Expected positive uniform density and nonnegative uniform viscosity.");
        }
    }
}
struct Derivatives { long double du_dx, du_dy, dv_dx, dv_dy; };
Derivatives derivatives(const FluidGrid& grid, size_t x, size_t y) {
    const auto i = static_cast<ptrdiff_t>(x), j = static_cast<ptrdiff_t>(y);
    const auto& left = grid.periodic_at(i - 1, j);
    const auto& right = grid.periodic_at(i + 1, j);
    const auto& down = grid.periodic_at(i, j - 1);
    const auto& up = grid.periodic_at(i, j + 1);
    return {(static_cast<long double>(right.velocity_x) - left.velocity_x) / (2.L * grid.dx()),
            (static_cast<long double>(up.velocity_x) - down.velocity_x) / (2.L * grid.dy()),
            (static_cast<long double>(right.velocity_y) - left.velocity_y) / (2.L * grid.dx()),
            (static_cast<long double>(up.velocity_y) - down.velocity_y) / (2.L * grid.dy())};
}

double reference_wave(const FluidGrid& grid, double time, const TaylorGreen& spec) {
    if (!isfinite(time) || time < 0 || !isfinite(spec.speed) || spec.speed <= 0 ||
        !isfinite(spec.density) || spec.density <= 0 || !isfinite(spec.viscosity) ||
        spec.viscosity < 0 || !spec.mode)
        throw invalid_argument("Invalid Taylor-Green time or material/amplitude/mode.");
    const auto& domain = grid.domain();
    if (abs(domain.length_x - domain.length_y) > 1e-12 * max(domain.length_x, domain.length_y) ||
        grid.width() / 4 < spec.mode || grid.height() / 4 < spec.mode)
        throw invalid_argument("Taylor-Green requires a square domain and >=4 cells per wavelength.");
    const double k = 2 * pi * spec.mode / domain.length_x;
    if (!isfinite(k) || !isfinite(spec.speed * spec.speed * spec.density))
        throw invalid_argument("Taylor-Green scale overflows.");
    return k;
}
double amplitude(double time, const TaylorGreen& spec, double k) {
    return spec.speed * exp(-2 * spec.viscosity * k * k * time);
}
long double relative(long double error_squared, long double reference_squared) {
    if (reference_squared == 0)
        return error_squared == 0 ? 0 : numeric_limits<long double>::infinity();
    return sqrt(error_squared / reference_squared);
}
}

FlowDiagnostics diagnose_flow(const FluidGrid& grid, double dt) {
    validate_state(grid);
    if (!isfinite(dt) || dt < 0) throw invalid_argument("Diagnostic dt must be finite and nonnegative.");
    FlowDiagnostics out;
    long double div2 = 0, curl2 = 0, energy = 0, sum_u = 0, sum_v = 0, max_courant = 0;
    for (size_t y = 0; y < grid.height(); ++y) {
        for (size_t x = 0; x < grid.width(); ++x) {
            const auto d = derivatives(grid, x, y);
            const auto& c = grid.at(x, y);
            const long double div = d.du_dx + d.dv_dy, curl = d.dv_dx - d.du_dy;
            div2 += div * div; curl2 += curl * curl;
            out.divergence_max = max(out.divergence_max, finite_result(abs(div)));
            out.max_speed = max(out.max_speed, finite_result(hypotl(c.velocity_x, c.velocity_y)));
            sum_u += c.velocity_x; sum_v += c.velocity_y;
            energy += (static_cast<long double>(c.velocity_x) * c.velocity_x +
                       static_cast<long double>(c.velocity_y) * c.velocity_y) / 2;
            max_courant = max(max_courant, dt * (abs(static_cast<long double>(c.velocity_x)) / grid.dx() +
                                               abs(static_cast<long double>(c.velocity_y)) / grid.dy()));
        }
    }
    const long double count = static_cast<long double>(grid.width()) * grid.height();
    out.divergence_rms = finite_result(sqrt(div2 / count));
    out.vorticity_rms = finite_result(sqrt(curl2 / count));
    out.mean_velocity_x = finite_result(sum_u / count);
    out.mean_velocity_y = finite_result(sum_v / count);
    out.kinetic_energy_per_mass = finite_result(energy / count);
    out.mass_per_depth = finite_result(static_cast<long double>(grid.at(0, 0).density) *
                                      grid.domain().length_x * grid.domain().length_y);
    out.advective_courant = finite_result(max_courant);
    out.diffusion_number = finite_result(static_cast<long double>(dt) * grid.at(0, 0).kinematic_viscosity *
        (1.L / grid.dx() / grid.dx() + 1.L / grid.dy() / grid.dy()));
    return out;
}

vector<double> vorticity_field(const FluidGrid& grid) {
    validate_state(grid);
    vector<double> values(grid.width() * grid.height());
    for (size_t y = 0; y < grid.height(); ++y)
        for (size_t x = 0; x < grid.width(); ++x) {
            const auto d = derivatives(grid, x, y);
            values[y * grid.width() + x] = finite_result(d.dv_dx - d.du_dy);
        }
    return values;
}

void fill_taylor_green(FluidGrid& grid, double time, const TaylorGreen& spec) {
    const double k = reference_wave(grid, time, spec);
    const double a = amplitude(time, spec, k);
    for (size_t y = 0; y < grid.height(); ++y) {
        const double phase_y = 2 * pi * spec.mode * (static_cast<double>(y) + 0.5) / grid.height();
        for (size_t x = 0; x < grid.width(); ++x) {
            const double phase_x = 2 * pi * spec.mode * (static_cast<double>(x) + 0.5) / grid.width();
            Cell c;
            c.velocity_x = a * sin(phase_x) * cos(phase_y);
            c.velocity_y = -a * cos(phase_x) * sin(phase_y);
            c.pressure = spec.density * a * a / 4 * (cos(2 * phase_x) + cos(2 * phase_y));
            c.density = spec.density; c.kinematic_viscosity = spec.viscosity;
            grid.at(x, y) = c;
        }
    }
}

ReferenceErrors compare_taylor_green(const FluidGrid& candidate, double time, const TaylorGreen& spec) {
    const auto metrics = diagnose_flow(candidate);
    const double k = reference_wave(candidate, time, spec);
    const double a = amplitude(time, spec, k);
    if (abs(candidate.at(0, 0).kinematic_viscosity - spec.viscosity) >
        material_tolerance * max(candidate.at(0, 0).kinematic_viscosity, spec.viscosity))
        throw invalid_argument("Benchmark viscosity differs from the reference.");
    for (size_t y = 0; y < candidate.height(); ++y)
        for (size_t x = 0; x < candidate.width(); ++x)
            if (candidate.at(x, y).force_x != 0 || candidate.at(x, y).force_y != 0)
                throw invalid_argument("Taylor-Green benchmark must be unforced.");
    FluidGrid reference(candidate.width(), candidate.height(), candidate.domain());
    fill_taylor_green(reference, time, spec);
    const long double count = static_cast<long double>(candidate.width()) * candidate.height();
    long double pressure_offset = 0;
    for (size_t y = 0; y < candidate.height(); ++y)
        for (size_t x = 0; x < candidate.width(); ++x)
            pressure_offset += static_cast<long double>(candidate.at(x, y).pressure) - reference.at(x, y).pressure;
    pressure_offset /= count;
    ReferenceErrors errors;
    long double uv2 = 0, ref_uv2 = 0, p2 = 0, omega2 = 0, ref_omega2 = 0;
    for (size_t y = 0; y < candidate.height(); ++y) {
        const double phase_y = 2 * pi * spec.mode * (static_cast<double>(y) + 0.5) / candidate.height();
        for (size_t x = 0; x < candidate.width(); ++x) {
            const auto& c = candidate.at(x, y); const auto& r = reference.at(x, y);
            const long double du = static_cast<long double>(c.velocity_x) - r.velocity_x;
            const long double dv = static_cast<long double>(c.velocity_y) - r.velocity_y;
            uv2 += du * du + dv * dv;
            ref_uv2 += static_cast<long double>(r.velocity_x) * r.velocity_x +
                       static_cast<long double>(r.velocity_y) * r.velocity_y;
            errors.velocity_max_absolute = max(errors.velocity_max_absolute, finite_result(hypotl(du, dv)));
            const long double dp = static_cast<long double>(c.pressure) - r.pressure - pressure_offset;
            p2 += dp * dp;
            const double phase_x = 2 * pi * spec.mode * (static_cast<double>(x) + 0.5) / candidate.width();
            const long double exact_curl = 2.L * k * a * sin(phase_x) * sin(phase_y);
            const auto d = derivatives(candidate, x, y);
            const long double error = d.dv_dx - d.du_dy - exact_curl;
            omega2 += error * error; ref_omega2 += exact_curl * exact_curl;
        }
    }
    errors.velocity_relative_l2 = static_cast<double>(relative(uv2, ref_uv2));
    errors.pressure_gauge_rms = finite_result(sqrt(p2 / count));
    errors.vorticity_relative_l2 = static_cast<double>(relative(omega2, ref_omega2));
    const long double expected_energy = static_cast<long double>(a) * a / 4;
    errors.energy_relative = static_cast<double>(relative(
        pow(static_cast<long double>(metrics.kinetic_energy_per_mass) - expected_energy, 2),
        expected_energy * expected_energy));
    const long double mass = static_cast<long double>(spec.density) * candidate.domain().length_x * candidate.domain().length_y;
    errors.mass_relative = finite_result(abs(metrics.mass_per_depth - mass) / mass);
    return errors;
}

ValidationResult validate_taylor_green_rollout(FluidStepper& stepper, size_t size, double dt,
    size_t steps, const TaylorGreen& spec, const ValidationLimits& limits) {
    if (!steps || steps == numeric_limits<size_t>::max() || !isfinite(dt) || dt <= 0 ||
        !isfinite(dt * static_cast<double>(steps)))
        throw invalid_argument("Rollout requires finite positive dt, steps and final time.");
    for (double limit : {limits.velocity_relative_l2, limits.divergence_scaled_max, limits.pressure_scaled_rms,
                        limits.vorticity_relative_l2, limits.energy_relative, limits.mass_relative})
        if (!isfinite(limit) || limit < 0) throw invalid_argument("Limits must be finite and nonnegative.");
    FluidGrid initial(size, size);
    fill_taylor_green(initial, 0, spec);
    Simulation simulation(move(initial));
    ValidationResult worst;
    worst.passed = true;
    for (size_t i = 0; i <= steps; ++i) {
        if (i) simulation.advance(stepper, dt);
        const auto d = diagnose_flow(simulation.state(), dt);
        const auto e = compare_taylor_green(simulation.state(), simulation.time(), spec);
        const bool passed = e.velocity_relative_l2 <= limits.velocity_relative_l2 &&
            d.divergence_max * simulation.state().domain().length_x / spec.speed <= limits.divergence_scaled_max &&
            e.pressure_gauge_rms / spec.density / spec.speed / spec.speed <= limits.pressure_scaled_rms &&
            e.vorticity_relative_l2 <= limits.vorticity_relative_l2 && e.energy_relative <= limits.energy_relative &&
            e.mass_relative <= limits.mass_relative;
        worst.passed = worst.passed && passed;
        worst.diagnostics = d;
        worst.worst_divergence_scaled = max(worst.worst_divergence_scaled,
            d.divergence_max * simulation.state().domain().length_x / spec.speed);
        worst.errors.velocity_relative_l2 = max(worst.errors.velocity_relative_l2, e.velocity_relative_l2);
        worst.errors.velocity_max_absolute = max(worst.errors.velocity_max_absolute, e.velocity_max_absolute);
        worst.errors.pressure_gauge_rms = max(worst.errors.pressure_gauge_rms, e.pressure_gauge_rms);
        worst.errors.vorticity_relative_l2 = max(worst.errors.vorticity_relative_l2, e.vorticity_relative_l2);
        worst.errors.energy_relative = max(worst.errors.energy_relative, e.energy_relative);
        worst.errors.mass_relative = max(worst.errors.mass_relative, e.mass_relative);
    }
    return worst;
}
