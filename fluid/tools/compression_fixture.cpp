// Synthetic benchmark inputs and exact readback; not a production loader/writer.
#include "FlowPresets.h"
#include "GridOutput.h"
#include "RunConfig.h"
#include "Validation.h"
#include <H5Cpp.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;
namespace {
const char* names[] = {"velocity_x", "velocity_y", "pressure", "density",
                       "kinematic_viscosity", "force_x", "force_y"};
vector<double> read_field(H5::H5File& file, const char* name) {
    auto data = file.openDataSet(string("/cells/") + name);
    if (data.getDataType() != H5::PredType::IEEE_F64LE) throw runtime_error("Unexpected fixture type.");
    const auto space = data.getSpace();
    if (space.getSimpleExtentNdims() != 2) throw runtime_error("Unexpected fixture rank.");
    hsize_t dims[2]; space.getSimpleExtentDims(dims);
    vector<double> values(dims[0] * dims[1]);
    data.read(values.data(), H5::PredType::NATIVE_DOUBLE);
    return values;
}
void generate(const string& path, size_t size, const string& profile) {
    FluidGrid grid(size, size);
    if (profile == "gradient") initialize_flow(grid, FlowPattern::Gradient);
    else if (profile == "taylor_green") fill_taylor_green(grid, 0.3);
    else if (profile == "multiscale" || profile == "noise") {
        fill_taylor_green(grid, 0.3);
        mt19937_64 rng(20261020);
        // Define our conversion instead of implementation-specific uniform_real_distribution.
        const auto noise = [&] { return 2.0 * static_cast<double>(rng() >> 11) * 0x1.0p-53 - 1.0; };
        const double pi = acos(-1.0);
        for (size_t y = 0; y < size; ++y) {
            const double py = 2 * pi * (static_cast<double>(y) + 0.5) / size;
            for (size_t x = 0; x < size; ++x) {
                auto& c = grid.at(x, y);
                const double px = 2 * pi * (static_cast<double>(x) + 0.5) / size;
                if (profile == "noise") {
                    c.velocity_x = noise(); c.velocity_y = noise(); c.pressure = noise();
                } else {
                    c.velocity_x = 0; c.velocity_y = 0; c.pressure = 0;
                    for (unsigned mode = 1; mode <= 16; mode *= 2) {
                        const double a = 1.0 / mode;
                        c.velocity_x += a * sin(mode * px + 0.31 * mode) * cos(mode * py + 0.17 * mode);
                        c.velocity_y -= a * cos(mode * px + 0.31 * mode) * sin(mode * py + 0.17 * mode);
                        c.pressure += a * cos(mode * px + mode * py);
                    }
                }
            }
        }
    } else throw invalid_argument("Unknown synthetic profile.");
    save_hdf5(grid, path);
}
void inspect(const string& path) {
    H5::H5File file(path, H5F_ACC_RDONLY);
    cout << '{'; bool first = true;
    for (const auto name : names) {
        auto data = file.openDataSet(string("/cells/") + name);
        if (!first) cout << ',';
        first = false;
        auto properties = data.getCreatePlist();
        cout << '"' << name << "\":{\"stored_bytes\":" << data.getStorageSize()
             << ",\"filter_count\":" << properties.getNfilters() << '}';
    }
    cout << "}\n";
}
}
int main(int argc, char** argv) {
    try {
        H5::Exception::dontPrint();
        const string action = argc > 1 ? argv[1] : "";
        if (action == "--generate" && argc == 5) generate(argv[2], parse_size(argv[3]), argv[4]);
        else if (action == "--inspect" && argc == 3) inspect(argv[2]);
        else if (action == "--read" && argc == 3) {
            H5::H5File file(argv[2], H5F_ACC_RDONLY);
            uint64_t hash = 14695981039346656037ULL;
            for (const auto name : names) {
                const auto values = read_field(file, name);
                for (double value : values) {
                    uint64_t bits; memcpy(&bits, &value, sizeof(bits));
                    hash = (hash ^ bits) * 1099511628211ULL;
                }
            }
            cout << hash << '\n';
        } else if (action == "--verify" && argc == 4) {
            H5::H5File original(argv[2], H5F_ACC_RDONLY), candidate(argv[3], H5F_ACC_RDONLY);
            for (const auto name : names) {
                const auto a = read_field(original, name), b = read_field(candidate, name);
                if (a.size() != b.size() || memcmp(a.data(), b.data(), a.size() * sizeof(double)))
                    throw runtime_error("Bitwise mismatch after repacking.");
            }
            cout << "bitwise_equal\n";
        } else throw invalid_argument("Use --generate PATH SIZE PROFILE, --read PATH, --inspect PATH, or --verify A B.");
        return 0;
    } catch (const H5::Exception& e) { cerr << e.getDetailMsg() << '\n'; }
      catch (const exception& e) { cerr << e.what() << '\n'; }
    return 1;
}
