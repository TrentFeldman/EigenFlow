#pragma once

#include "RunConfig.h"
#include "Simulation.h"

using namespace std;

// Make a new YYYYMMDD-name[-NNN] folder under output_root, relative to cwd.
// Existing folders are never reused, including failed/interrupted runs.
// Write resolved.cfg, environment.txt and a running status; I/O errors propagate.
filesystem::path begin_run(const RunConfig& config);

// Overwrite only this run's status file. Completed means initialization/export
// succeeded, never that a physical simulation has been performed.
enum class RunStatus { Running, Initialized, Failed };
void record_status(const filesystem::path& directory, RunStatus status);

// Create the t=0 state and export into a fresh run folder, returning that path.
// Failure preserves the folder and marks it failed when possible, then rethrows.
filesystem::path export_initial_state(const RunConfig& config);
