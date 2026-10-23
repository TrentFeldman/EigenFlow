// Analytical checks establish the validation harness, not a numerical CFD solver.
#include "Validation.h"
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace std;
namespace {
const double pi = acos(-1.0);
void require(bool passed, const char* message) { if (!passed) throw runtime_error(message); }
void near(double a, double b, double tolerance = 1e-12) {
    require(isfinite(a) && abs(a - b) <= tolerance, "Numeric check failed.");
}
void rejects(const function<void()>& call) {
    try { call(); } catch (const invalid_argument&) { return; }
    throw runtime_error("Expected input rejection.");
}
// This test double evaluates the exact formula; it does no numerical integration.
class ExactFormula : public FluidStepper {
public:
    void step(const FluidGrid&, FluidGrid& next, double time, double dt) override {
        fill_taylor_green(next, time + dt);
    }
};
class FrozenFlow : public FluidStepper {
public:
    void step(const FluidGrid&, FluidGrid&, double, double) override {}
};
class TransientError : public FluidStepper {
public:
    void step(const FluidGrid&, FluidGrid& next, double time, double dt) override {
        fill_taylor_green(next, time + dt);
        if (time < 0.05) next.at(0, 0).velocity_x += 1;
    }
};
void constant_flow() {
    FluidGrid grid(12, 8, {6, 4, 10, -2});
    for (size_t y = 0; y < grid.height(); ++y)
        for (size_t x = 0; x < grid.width(); ++x) {
            auto& c = grid.at(x, y);
            c.velocity_x = 2; c.velocity_y = -1;
            c.density = 3; c.kinematic_viscosity = 0.1;
        }
    const auto d = diagnose_flow(grid, 0.25);
    near(d.divergence_max, 0); near(d.vorticity_rms, 0);
    near(d.kinetic_energy_per_mass, 2.5); near(d.mass_per_depth, 72);
    near(d.advective_courant, 1.5); near(d.diffusion_number, 0.2);
    near(d.mean_velocity_x, 2); near(d.mean_velocity_y, -1);
    for (double curl : vorticity_field(grid)) near(curl, 0);
    for (double Cell::*field : {&Cell::velocity_x, &Cell::velocity_y, &Cell::pressure,
            &Cell::density, &Cell::kinematic_viscosity, &Cell::force_x, &Cell::force_y}) {
        const auto saved = grid.at(1, 1).*field;
        grid.at(1, 1).*field = numeric_limits<double>::quiet_NaN();
        rejects([&] { diagnose_flow(grid); });
        grid.at(1, 1).*field = saved;
    }
    grid.at(1, 1).density = 4;
    rejects([&] { diagnose_flow(grid); });
    grid.at(1, 1).density = 3;
    grid.at(1, 1).kinematic_viscosity = -1;
    rejects([&] { diagnose_flow(grid); });
    rejects([] { diagnose_flow(FluidGrid(2, 8)); });
    rejects([] { diagnose_flow(FluidGrid(8, 8), -1); });
}
void reference_and_convergence() {
    TaylorGreen spec;
    double previous_error = 0;
    cout << "size,curl_relative_error,observed_order,max_divergence,energy_relative_error\n";
    for (size_t size : {16, 32, 64, 128}) {
        FluidGrid grid(size, size);
        fill_taylor_green(grid, 0.5, spec);
        const auto d = diagnose_flow(grid);
        const auto e = compare_taylor_green(grid, 0.5, spec);
        const double h = 2 * pi / static_cast<double>(size);
        near(e.vorticity_relative_l2, 1 - sin(h) / h);
        require(d.divergence_max < 1e-11, "Reference divergence.");
        near(e.velocity_relative_l2, 0); near(e.energy_relative, 0);
        const double order = previous_error ? log2(previous_error / e.vorticity_relative_l2) : 0;
        if (previous_error) require(order > 1.95 && order < 2.05, "Curl stencil failed second-order refinement.");
        cout << size << ',' << e.vorticity_relative_l2 << ',' << order << ','
             << d.divergence_max << ',' << e.energy_relative << '\n';
        previous_error = e.vorticity_relative_l2;
    }
    FluidGrid grid(64, 64);
    fill_taylor_green(grid, 0);
    const auto start_energy = diagnose_flow(grid).kinetic_energy_per_mass;
    fill_taylor_green(grid, 1);
    near(diagnose_flow(grid).kinetic_energy_per_mass / start_energy,
         exp(-4 * spec.viscosity * pi * pi));
    for (size_t y = 0; y < grid.height(); ++y)
        for (size_t x = 0; x < grid.width(); ++x) grid.at(x, y).pressure += 42;
    require(compare_taylor_green(grid, 1).pressure_gauge_rms < 1e-13, "Pressure gauge handling.");
    grid.at(2, 3).pressure += 0.1;
    require(compare_taylor_green(grid, 1).pressure_gauge_rms > 0.001, "Pressure error not detected.");
    fill_taylor_green(grid, 1);
    grid.at(2, 3).velocity_x += 0.1;
    require(compare_taylor_green(grid, 1).velocity_relative_l2 > 0.001, "Velocity error not detected.");
    rejects([] { FluidGrid g(16, 16, {2, 3, 0, 0}); fill_taylor_green(g, 0); });
    rejects([] { FluidGrid g(16, 16); fill_taylor_green(g, -1); });
    rejects([] { FluidGrid g(4, 4); fill_taylor_green(g, 0, {1, 1, 0.01, 2}); });
    rejects([] { FluidGrid g(16, 16); fill_taylor_green(g, 0, {0, 1, 0.01, 1}); });
    fill_taylor_green(grid, 0);
    grid.at(0, 0).force_x = 1;
    rejects([&] { compare_taylor_green(grid, 0); });
}
// Numerical derivatives of the reference itself check pressure sign and all
// terms of the continuous momentum equation, independently of discrete curl.
void momentum_residual() {
    const auto value = [](double x, double y, double t) {
        const double a = exp(-2 * 0.01 * pi * pi * t);
        return Cell{a * sin(pi * x) * cos(pi * y), -a * cos(pi * x) * sin(pi * y),
            a * a / 4 * (cos(2 * pi * x) + cos(2 * pi * y))};
    };
    const double h = 1e-4;
    for (const double x : {0.17, 0.41, 0.78}) {
        const double y = 0.33, t = 0.4;
        const auto c = value(x, y, t), l = value(x-h, y, t), r = value(x+h, y, t);
        const auto b = value(x, y-h, t), a = value(x, y+h, t);
        const auto past = value(x, y, t-h), future = value(x, y, t+h);
        const double u_residual = (future.velocity_x-past.velocity_x)/(2*h) +
            c.velocity_x*(r.velocity_x-l.velocity_x)/(2*h) +
            c.velocity_y*(a.velocity_x-b.velocity_x)/(2*h) + (r.pressure-l.pressure)/(2*h) -
            0.01*(r.velocity_x+l.velocity_x+a.velocity_x+b.velocity_x-4*c.velocity_x)/(h*h);
        const double v_residual = (future.velocity_y-past.velocity_y)/(2*h) +
            c.velocity_x*(r.velocity_y-l.velocity_y)/(2*h) +
            c.velocity_y*(a.velocity_y-b.velocity_y)/(2*h) + (a.pressure-b.pressure)/(2*h) -
            0.01*(r.velocity_y+l.velocity_y+a.velocity_y+b.velocity_y-4*c.velocity_y)/(h*h);
        require(abs(u_residual) < 1e-6 && abs(v_residual) < 1e-6, "Continuous momentum balance.");
    }
    // One independently known point anchors the exported reference parameterization.
    FluidGrid grid(8, 8, {2, 2, 0, 0});
    fill_taylor_green(grid, 0);
    near(grid.at(0, 0).velocity_x, sqrt(2.) / 4);
    near(grid.at(0, 0).velocity_y, -sqrt(2.) / 4);
    near(grid.at(0, 0).pressure, sqrt(2.) / 4);
}
void harness() {
    ExactFormula exact;
    const auto good = validate_taylor_green_rollout(exact, 64, 0.1, 10);
    require(good.passed, "Exact-formula test double should pass harness.");
    FrozenFlow frozen;
    require(!validate_taylor_green_rollout(frozen, 64, 0.1, 10).passed, "Frozen flow should fail decay.");
    TransientError transient;
    require(!validate_taylor_green_rollout(transient, 64, 0.1, 10).passed,
            "Intermediate error must fail even if final frame is exact.");
    ValidationLimits bad;
    bad.velocity_relative_l2 = -1;
    rejects([&] { validate_taylor_green_rollout(exact, 64, 0.1, 10, {}, bad); });
    rejects([&] { validate_taylor_green_rollout(exact, 64, 0.1, 0); });
    cout << "harness=passed; numerical_solver_validation=not_run (no solver)\n";
}
}
int main() {
    try {
        cout << setprecision(17);
        constant_flow(); reference_and_convergence(); momentum_residual(); harness();
        return 0;
    } catch (const exception& e) { cerr << e.what() << '\n'; return 1; }
}
