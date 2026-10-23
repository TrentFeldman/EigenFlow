// Reproducible output folders and explicit initialization-only run status.
#include "RunOutput.h"
#include "BuildInfo.h"
#include "GridOutput.h"
#include <H5Cpp.h>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;

void record_status(const filesystem::path& directory, RunStatus status) {
    const char* label = nullptr;
    switch (status) {
    case RunStatus::Running: label = "running"; break;
    case RunStatus::Initialized: label = "initialized"; break;
    case RunStatus::Failed: label = "failed"; break;
    default: throw invalid_argument("Unknown run status.");
    }
    ofstream out(directory / "status.txt");
    out << "status=" << label << "\nmode=initialization_only\n"
        << "solver=not_implemented\ntime=0\nstep=0\nseed=not_used\n";
    out.close();
    if (!out) throw runtime_error("Cannot record run status in " + directory.string());
}

filesystem::path begin_run(const RunConfig& config) {
    config.validate();
    const time_t now = std::time(nullptr);
    const tm* utc = std::gmtime(&now);
    if (!utc) throw runtime_error("Cannot determine UTC date.");
    ostringstream date;
    date << put_time(utc, "%Y%m%d");
    const string base = date.str() + "-" + config.name;
    filesystem::create_directories(config.output_root);
    filesystem::path directory;
    for (size_t suffix = 0;; ++suffix) {
        ostringstream name;
        name << base;
        if (suffix) name << '-' << setw(3) << setfill('0') << suffix;
        directory = config.output_root / name.str();
        if (filesystem::create_directory(directory)) break;
        if (suffix == numeric_limits<size_t>::max())
            throw runtime_error("No available run folder name.");
    }
    try {
        record_status(directory, RunStatus::Running);
        write_config(config, directory / "resolved.cfg");
        unsigned major{}, minor{}, release{};
        if (H5get_libversion(&major, &minor, &release) < 0)
            throw runtime_error("Cannot read HDF5 version.");
        ofstream environment(directory / "environment.txt");
        environment << "compiler=" << FLUID_COMPILER << "\nbuild_type=" << FLUID_BUILD_TYPE
                    << "\ncxx_standard=17\nhdf5_runtime=" << major << '.' << minor << '.' << release
                    << "\nsource_sha256=" << FLUID_SOURCE_SHA256
                    << "\nworking_directory=" << filesystem::current_path().string()
                    << "\ndate_basis=UTC\n";
        environment.close();
        if (!environment) throw runtime_error("Cannot write environment metadata.");
    } catch (...) {
        try { record_status(directory, RunStatus::Failed); } catch (...) {}
        throw;
    }
    return directory;
}

filesystem::path export_initial_state(const RunConfig& config) {
    const auto directory = begin_run(config);
    try {
        FluidGrid grid(config.width, config.height, config.domain);
        initialize_flow(grid, config.pattern, config.density, config.viscosity);
        const Simulation simulation(move(grid));
        print_grid(simulation.state());
        save_hdf5(simulation.state(), (directory / "grid.h5").string(),
                   simulation.time(), simulation.step_count());
        save_svg(simulation.state(), (directory / "grid.svg").string());
        record_status(directory, RunStatus::Initialized);
    } catch (...) {
        try { record_status(directory, RunStatus::Failed); } catch (...) {}
        throw;
    }
    return directory;
}
