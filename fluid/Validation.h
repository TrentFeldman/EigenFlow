#pragma once

#include "Simulation.h"
#include <vector>

using namespace std;

// Independent periodic diagnostics for cell-centered [height][width] data.
struct FlowDiagnostics {
    double divergence_rms = 0;       // 1/s; du/dx + dv/dy.
    double divergence_max = 0;
    double vorticity_rms = 0;        // 1/s; dv/dx - du/dy.
    double max_speed = 0;            // m/s.
    double mean_velocity_x = 0;      // m/s; tracks mean momentum at fixed density.
    double mean_velocity_y = 0;
    double kinetic_energy_per_mass = 0; // m^2/s^2; mean((u^2+v^2)/2).
    double mass_per_depth = 0;       // kg/m; 2D domain assumes unit depth.
    double advective_courant = 0;    // max(dt*(abs(u)/dx + abs(v)/dy)).
    double diffusion_number = 0;     // nu*dt*(1/dx^2 + 1/dy^2).
};

// Second-order centered derivatives, wrapped neighbors. Require >=3 cells/axis,
// finite fields, uniform positive density and uniform nonnegative viscosity
// (relative material tolerance 1e-12). dt >= 0 in seconds. Invalid inputs throw
// invalid_argument; unrepresentable diagnostics throw overflow_error.
FlowDiagnostics diagnose_flow(const FluidGrid& grid, double dt = 0);
// Same stencil/validation; return scalar curl in flat row-major order [y*width+x].
vector<double> vorticity_field(const FluidGrid& grid);

struct TaylorGreen {
    double speed = 1.0;       // Initial velocity amplitude, m/s; positive.
    double density = 1.0;     // kg/m^3.
    double viscosity = 0.01;  // m^2/s, >=0.
    unsigned mode = 1;        // Whole periods per axis, positive.
};

// Exact unforced Navier-Stokes reference on a square periodic domain. Samples
// cell centers at time >=0 (seconds), resets all fields and uses zero-mean gauge
// pressure. Require >=4 cells per wavelength to avoid degenerate sampling.
// This evaluates a formula; it does not numerically integrate a simulation.
void fill_taylor_green(FluidGrid& grid, double time, const TaylorGreen& spec = {});

struct ReferenceErrors {
    double velocity_relative_l2 = 0;
    double velocity_max_absolute = 0;  // m/s; Euclidean vector error.
    double pressure_gauge_rms = 0;     // Pa; mean pressure offset removed.
    double vorticity_relative_l2 = 0;   // Centered discrete curl vs exact curl.
    double energy_relative = 0;
    double mass_relative = 0;
};

// Compare candidate with the exact reference at the same time/geometry. A zero
// reference norm gives 0 only for zero error, infinity otherwise. Pressure gauge
// offsets are ignored. Vorticity includes diagnostic stencil truncation error.
ReferenceErrors compare_taylor_green(const FluidGrid& candidate, double time,
                                    const TaylorGreen& spec = {});

// Provisional reference-solver gates, distinct from the ML model's 5% target.
struct ValidationLimits {
    double velocity_relative_l2 = 0.01;
    double divergence_scaled_max = 1e-6; // max(abs(div))*domain_length/initial_speed.
    double pressure_scaled_rms = 0.02;  // Gauge RMS/(rho*initial_speed^2).
    double vorticity_relative_l2 = 0.02;
    double energy_relative = 0.02;
    double mass_relative = 1e-12;
};
struct ValidationResult {
    FlowDiagnostics diagnostics; // Final frame.
    ReferenceErrors errors;
    double worst_divergence_scaled = 0;
    bool passed = false;
};

// Connects a real stepper when one exists. Starts exact at t=0, runs fixed dt
// steps, checks every completed frame, and accumulates worst errors/divergence.
// Invalid setup/step failures propagate; tolerance failures set passed=false.
// No production FluidStepper is supplied yet; a passing exact-formula test
// double establishes harness behavior only, never numerical solver validation.
ValidationResult validate_taylor_green_rollout(FluidStepper& stepper, size_t size,
    double dt, size_t steps, const TaylorGreen& spec = {}, const ValidationLimits& limits = {});
