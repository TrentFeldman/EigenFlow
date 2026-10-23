#pragma once

#include "FlowPresets.h"
#include <filesystem>
#include <string>
#include <string_view>

using namespace std;

// Complete initialization recipe. dt/duration are future solver settings in
// seconds; the current executable only exports t=0. No RNG is used.
struct RunConfig {
    size_t width = 64;
    size_t height = 64;
    Domain domain;
    FlowPattern pattern = FlowPattern::Gradient;
    double density = 1.0;
    double viscosity = 0.01;
    double dt = 0.001;
    double duration = 1.0;
    string name = "initialization";
    filesystem::path output_root = "saves";

    // Validate without allocating a grid. Invalid values throw invalid_argument.
    void validate() const;
};

// Strict unsigned decimal CLI/config parsing; no signs, overflow or trailing text.
size_t parse_size(string_view text);

// Flat key=value file; # comments, blank lines and surrounding spaces allowed.
// Missing keys use defaults; duplicate/unknown keys and bad values are errors.
RunConfig read_config(const filesystem::path& path);
// Write all resolved values, sufficient to repeat initialization. Overwrites path.
// File errors throw runtime_error; parsing/validation errors throw invalid_argument.
void write_config(const RunConfig& config, const filesystem::path& path);
