#pragma once

#include "FluidGrid.h"
#include <cstdint>

using namespace std;

// Interface for future physics. No Navier–Stokes implementation exists yet.
class FluidStepper {
public:
    virtual ~FluidStepper() = default;
    // Read current, write next (same geometry), using time and dt in seconds.
    // next starts as a copy of current, so unchanged properties persist.
    // A failure must throw; do not retain references to either buffer.
    virtual void step(const FluidGrid& current, FluidGrid& next, double time, double dt) = 0;
};

// Owns two independent buffers and advances time only after a successful step.
class Simulation {
public:
    explicit Simulation(FluidGrid initial);
    const FluidGrid& state() const noexcept;
    double time() const noexcept;
    uint64_t step_count() const noexcept;

    // dt must be positive/finite and advance representable time.
    // Exceptions leave current state/time/count unchanged; next is reset on retry.
    // This manages buffers, not physical validity or CFL stability.
    void advance(FluidStepper& stepper, double dt);
private:
    FluidGrid current_;
    FluidGrid next_;
    double time_ = 0.0;
    uint64_t step_count_ = 0;
};
