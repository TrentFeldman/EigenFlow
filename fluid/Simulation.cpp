// Update ownership and clock bookkeeping, independent of the numerical method.
#include "Simulation.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace std;

Simulation::Simulation(FluidGrid initial) : current_(move(initial)), next_(current_) {}
const FluidGrid& Simulation::state() const noexcept { return current_; }
double Simulation::time() const noexcept { return time_; }
uint64_t Simulation::step_count() const noexcept { return step_count_; }

void Simulation::advance(FluidStepper& stepper, double dt) {
    const double next_time = time_ + dt;
    if (!isfinite(dt) || dt <= 0 || !isfinite(next_time) || next_time <= time_ ||
        step_count_ == numeric_limits<uint64_t>::max())
        throw invalid_argument("Step must advance a finite clock and representable step count.");
    next_ = current_;
    stepper.step(current_, next_, time_, dt);
    if (!current_.same_geometry(next_))
        throw invalid_argument("Stepper changed the grid geometry.");
    swap(current_, next_);
    time_ = next_time;
    ++step_count_;
}
