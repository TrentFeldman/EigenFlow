// Dependency-free configuration parsing and exact double-precision round trips.
#include "RunConfig.h"
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <set>
#include <stdexcept>

using namespace std;

namespace {
string trim(string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == string::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

double parse_real(const string& text) {
    double value{};
    const auto result = from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != errc{} || result.ptr != text.data() + text.size() || !isfinite(value))
        throw invalid_argument("Expected a finite decimal number: " + text);
    return value;
}
}

size_t parse_size(string_view text) {
    size_t value{};
    const auto result = from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != errc{} || result.ptr != text.data() + text.size())
        throw invalid_argument("Expected an unsigned decimal integer: " + string(text));
    return value;
}

void RunConfig::validate() const {
    domain.validate();
    const auto limit = static_cast<size_t>(numeric_limits<ptrdiff_t>::max());
    if (!width || !height || width > limit || height > limit ||
        width > numeric_limits<size_t>::max() / height / sizeof(Cell) ||
        !(domain.length_x / static_cast<double>(width) > 0) ||
        !(domain.length_y / static_cast<double>(height) > 0))
        throw invalid_argument("Invalid grid dimensions or spacing.");
    (void)pattern_name(pattern);
    if (!isfinite(density) || density <= 0 || !isfinite(viscosity) || viscosity < 0)
        throw invalid_argument("Density must be positive and viscosity nonnegative, both finite.");
    if (!isfinite(dt) || dt <= 0 || !isfinite(duration) || duration <= 0 || dt > duration)
        throw invalid_argument("Require finite times with 0 < dt <= duration.");
    if (name.empty() || name.size() > 64 ||
        name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
            != string::npos)
        throw invalid_argument("Run name must contain 1-64 letters, digits, hyphens or underscores.");
    const string root = output_root.string();
    if (root.empty() || root != trim(root) || root.find_first_of("#\r\n") != string::npos)
        throw invalid_argument("Output root must be nonempty without #, newlines or surrounding spaces.");
}

RunConfig read_config(const filesystem::path& path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open config: " + path.string());
    RunConfig config;
    set<string> seen;
    string line;
    size_t number = 0;
    while (getline(in, line)) {
        ++number;
        const auto comment = line.find('#');
        if (comment != string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty()) continue;
        try {
            const auto separator = line.find('=');
            if (separator == string::npos) throw invalid_argument("Expected key=value.");
            const auto key = trim(line.substr(0, separator));
            const auto value = trim(line.substr(separator + 1));
            if (value.empty() || !seen.insert(key).second)
                throw invalid_argument("Empty value or duplicate key: " + key);
            if (key == "width") config.width = parse_size(value);
            else if (key == "height") config.height = parse_size(value);
            else if (key == "length_x") config.domain.length_x = parse_real(value);
            else if (key == "length_y") config.domain.length_y = parse_real(value);
            else if (key == "origin_x") config.domain.origin_x = parse_real(value);
            else if (key == "origin_y") config.domain.origin_y = parse_real(value);
            else if (key == "pattern") config.pattern = parse_pattern(value);
            else if (key == "density") config.density = parse_real(value);
            else if (key == "viscosity") config.viscosity = parse_real(value);
            else if (key == "dt") config.dt = parse_real(value);
            else if (key == "duration") config.duration = parse_real(value);
            else if (key == "name") config.name = value;
            else if (key == "output_root") config.output_root = value;
            else if (key == "boundary_x" || key == "boundary_y") {
                if (value != "periodic") throw invalid_argument("Only periodic boundaries are supported.");
            } else if (key == "field_location") {
                if (value != "cell_center") throw invalid_argument("Only cell_center storage is supported.");
            } else throw invalid_argument("Unknown key: " + key);
        } catch (const invalid_argument& error) {
            throw invalid_argument(path.string() + ":" + to_string(number) + ": " + error.what());
        }
    }
    if (!in.eof()) throw runtime_error("Failed while reading config: " + path.string());
    config.validate();
    return config;
}

void write_config(const RunConfig& config, const filesystem::path& path) {
    config.validate();
    ofstream out(path);
    out.imbue(locale::classic());
    out.precision(numeric_limits<double>::max_digits10);
    out << "# Resolved initialization recipe; SI units. No RNG.\n"
        << "width=" << config.width << "\nheight=" << config.height
        << "\nlength_x=" << config.domain.length_x << "\nlength_y=" << config.domain.length_y
        << "\norigin_x=" << config.domain.origin_x << "\norigin_y=" << config.domain.origin_y
        << "\nboundary_x=periodic\nboundary_y=periodic\nfield_location=cell_center"
        << "\npattern=" << pattern_name(config.pattern) << "\ndensity=" << config.density
        << "\nviscosity=" << config.viscosity << "\ndt=" << config.dt
        << "\nduration=" << config.duration << "\nname=" << config.name
        << "\noutput_root=" << config.output_root.string() << '\n';
    out.close();
    if (!out) throw runtime_error("Cannot write config: " + path.string());
}
