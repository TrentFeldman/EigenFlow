// Repeatable software checks; synthetic steppers below are not CFD models.
#include "RunOutput.h"
#include "GridOutput.h"
#include <H5Cpp.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace std;
namespace fs = filesystem;

namespace {
void check(bool condition, const string& message) {
    if (!condition) throw runtime_error(message);
}
void near(double actual, double expected) {
    check(abs(actual - expected) <= 1e-12, "Numeric mismatch.");
}
template<class Exception = invalid_argument, class Function>
void rejects(Function action) {
    try { action(); } catch (const Exception&) { return; }
    throw runtime_error("Expected exception was not raised.");
}
string read_text(const fs::path& path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot read test artifact.");
    return {istreambuf_iterator<char>(in), istreambuf_iterator<char>()};
}
void write_text(const fs::path& path, const string& text) {
    ofstream out(path);
    out << text;
    out.close();
    check(bool(out), "Cannot write test artifact.");
}
string text_attribute(const H5::H5Object& object, const char* name) {
    const auto attribute = object.openAttribute(name);
    string value;
    attribute.read(attribute.getStrType(), value);
    return value;
}
double real_attribute(const H5::H5Object& object, const char* name) {
    double value{};
    object.openAttribute(name).read(H5::PredType::NATIVE_DOUBLE, &value);
    return value;
}
uint64_t integer_attribute(const H5::H5Object& object, const char* name) {
    uint64_t value{};
    object.openAttribute(name).read(H5::PredType::NATIVE_UINT64, &value);
    return value;
}

void geometry() {
    FluidGrid grid(4, 2, {8, 6, 10, -3});
    near(grid.dx(), 2); near(grid.dy(), 3);
    near(grid.x_center(0), 11); near(grid.x_center(3), 17);
    near(grid.y_center(0), -1.5); near(grid.y_center(1), 1.5);
    grid.at(3, 1).pressure = 42;
    check(&grid.periodic_at(-1, -1) == &grid.at(3, 1), "Negative wrap.");
    check(&grid.periodic_at(7, 5) == &grid.at(3, 1), "Multiple-period wrap.");
    check(&grid.periodic_at(-9, -5) == &grid.at(3, 1), "Negative multi-period wrap.");
    check(&grid.periodic_at(4, 2) == &grid.at(0, 0), "Positive seam.");
    const FluidGrid& view = grid;
    near(view.periodic_at(-1, -1).pressure, 42);
    near(view.at(0, 0).density, 1);
    FluidGrid single(1, 1);
    near(single.x_center(0), 0); near(single.y_center(0), 0);
    check(&single.periodic_at(numeric_limits<ptrdiff_t>::min(),
                              numeric_limits<ptrdiff_t>::max()) == &single.at(0, 0),
          "Single-cell extreme-index wrapping.");
    rejects<out_of_range>([&] { grid.at(4, 0); });
    rejects<out_of_range>([&] { view.at(0, 2); });
    rejects<out_of_range>([&] { grid.x_center(4); });
    rejects<out_of_range>([&] { grid.y_center(2); });
    rejects([] { FluidGrid invalid(0, 2); });
    rejects([] { FluidGrid invalid(2, 0); });
    rejects([] { FluidGrid invalid(numeric_limits<size_t>::max(), 2); });
    rejects([] { FluidGrid invalid(2, 2, {0, 2, 0, 0}); });
    rejects([] { FluidGrid invalid(2, 2, {2, -1, 0, 0}); });
    rejects([] { FluidGrid invalid(2, 2, {2, 2, INFINITY, 0}); });
    rejects([] { FluidGrid invalid(2, 2, {1, 2, 1e300, 0}); });
    rejects([] { FluidGrid invalid(2, 2, {numeric_limits<double>::denorm_min(), 2, 0, 0}); });
}

void presets() {
    for (const auto& shape : {pair<size_t,size_t>{4, 2}, {5, 3}, {1, 4}, {3, 1}, {1, 1}}) {
        FluidGrid outward(shape.first, shape.second), inward = outward, vortex = outward;
        initialize_flow(outward, FlowPattern::Center, 2, 0.02);
        initialize_flow(inward, FlowPattern::Inward);
        initialize_flow(vortex, FlowPattern::Vortex);
        for (size_t y = 0; y < outward.height(); ++y) {
            for (size_t x = 0; x < outward.width(); ++x) {
                const auto& c = outward.at(x, y);
                near(inward.at(x, y).velocity_x, -c.velocity_x);
                near(inward.at(x, y).velocity_y, -c.velocity_y);
                near(vortex.at(x, y).velocity_x, -c.velocity_y);
                near(vortex.at(x, y).velocity_y, c.velocity_x);
                near(c.density, 2); near(c.kinematic_viscosity, 0.02);
                near(c.pressure, 0); near(c.force_x, 0); near(c.force_y, 0);
            }
        }
        initialize_flow(outward, FlowPattern::Gradient);
        for (size_t y = 1; y < outward.height(); ++y)
            check(outward.at(0, y).velocity_x > outward.at(0, y - 1).velocity_x,
                  "Gradient must increase upward.");
        initialize_flow(outward, FlowPattern::Still);
        for (size_t y = 0; y < outward.height(); ++y)
            for (size_t x = 0; x < outward.width(); ++x) {
                near(outward.at(x, y).velocity_x, 0); near(outward.at(x, y).velocity_y, 0);
            }
    }
    FluidGrid grid(4, 2);
    initialize_flow(grid, FlowPattern::Center);
    near(grid.at(0, 0).velocity_x, -0.75); near(grid.at(0, 0).velocity_y, -0.5);
    initialize_flow(grid, FlowPattern::Gradient);
    near(grid.at(0, 0).velocity_x, 0.25); near(grid.at(0, 1).velocity_x, 0.75);
    for (const string name : {"still", "gradient", "center", "inward", "vortex"})
        check(pattern_name(parse_pattern(name)) == name, "Pattern naming round trip.");
    rejects([&] { initialize_flow(grid, static_cast<FlowPattern>(99)); });
    rejects([&] { initialize_flow(grid, FlowPattern::Still, 0); });
    rejects([&] { initialize_flow(grid, FlowPattern::Still, 1, -0.01); });
    rejects([&] { initialize_flow(grid, FlowPattern::Still, NAN); });
    rejects([] { parse_pattern("unknown"); });
}

// Deliberately artificial updates that expose in-place aliasing and stale buffers.
class ShiftStepper : public FluidStepper {
public:
    void step(const FluidGrid& current, FluidGrid& next, double, double) override {
        check(&current.at(0, 0) != &next.at(0, 0), "Buffers alias.");
        for (size_t x = 0; x < current.width(); ++x)
            next.at(x, 0).velocity_x = current.periodic_at(static_cast<ptrdiff_t>(x) - 1, 0).velocity_x;
    }
};
class FailingStepper : public FluidStepper {
public:
    void step(const FluidGrid&, FluidGrid& next, double, double) override {
        next.at(0, 0).density = 999;
        throw runtime_error("Synthetic failure.");
    }
};
class ResizeStepper : public FluidStepper {
public:
    void step(const FluidGrid&, FluidGrid& next, double, double) override { next = FluidGrid(1, 1); }
};
void simulation() {
    FluidGrid grid(3, 1);
    for (size_t x = 0; x < 3; ++x) grid.at(x, 0).velocity_x = static_cast<double>(x + 1);
    Simulation simulation(grid);
    ShiftStepper shift;
    simulation.advance(shift, 0.25);
    near(simulation.state().at(0, 0).velocity_x, 3);
    near(simulation.state().at(1, 0).velocity_x, 1);
    near(simulation.state().at(2, 0).velocity_x, 2);
    near(grid.at(0, 0).velocity_x, 1);
    FailingStepper failure;
    rejects<runtime_error>([&] { simulation.advance(failure, 0.25); });
    near(simulation.state().at(0, 0).density, 1);
    near(simulation.time(), 0.25);
    check(simulation.step_count() == 1, "Failed update changed clock.");
    ResizeStepper resize;
    rejects([&] { simulation.advance(resize, 0.25); });
    check(simulation.state().width() == 3, "Failed geometry update persisted.");
    simulation.advance(shift, 0.25);
    near(simulation.state().at(0, 0).velocity_x, 2);
    near(simulation.state().at(0, 0).density, 1);
    near(simulation.time(), 0.5);
    check(simulation.step_count() == 2, "Completed update count.");
    for (double dt : {0.0, -1.0, numeric_limits<double>::infinity(),
                      numeric_limits<double>::quiet_NaN(), numeric_limits<double>::denorm_min()})
        rejects([&] { simulation.advance(shift, dt); });
    near(simulation.time(), 0.5);
}

void config(const fs::path& directory) {
    const auto path = directory / "recipe.cfg";
    RunConfig original;
    original.width = 7; original.height = 3;
    original.domain = {3.75, 9.5, -2, 4};
    original.pattern = FlowPattern::Inward;
    original.viscosity = nextafter(0.01, 1);
    original.density = 3.25; original.dt = 0.123456789012345;
    original.duration = 5.125; original.name = "replay-test";
    original.output_root = "path with spaces";
    write_config(original, path);
    const auto loaded = read_config(path);
    write_config(loaded, directory / "copy.cfg");
    check(read_text(path) == read_text(directory / "copy.cfg"), "Resolved config changed on replay.");
    check(loaded.viscosity == original.viscosity, "Double precision lost in config.");
    FluidGrid first(original.width, original.height, original.domain);
    FluidGrid second(loaded.width, loaded.height, loaded.domain);
    initialize_flow(first, original.pattern, original.density, original.viscosity);
    initialize_flow(second, loaded.pattern, loaded.density, loaded.viscosity);
    check(first.same_geometry(second), "Replay geometry differs.");
    for (size_t y = 0; y < first.height(); ++y)
        for (size_t x = 0; x < first.width(); ++x) {
            near(first.at(x, y).velocity_x, second.at(x, y).velocity_x);
            near(first.at(x, y).velocity_y, second.at(x, y).velocity_y);
        }
    write_text(path, " # comment\n width = 4 # columns\nheight=2\n");
    check(read_config(path).width == 4, "Whitespace/comments not parsed.");
    for (const string bad : {"width=0", "height=-1", "width=4x", "width=4\nwidth=5",
                            "typo=1", "density=0", "viscosity=-1", "dt=nan", "duration=0",
                            "dt=2\nduration=1", "origin_x=inf", "length_y=0", "name=../bad",
                            "boundary_x=wall", "field_location=face", "pattern=oops", "no_equals",
                            "width=", "width=999999999999999999999999999999"}) {
        write_text(path, bad);
        rejects([&] { read_config(path); });
    }
    rejects<runtime_error>([&] { read_config(directory / "missing.cfg"); });
    rejects<runtime_error>([&] { write_config(original, directory / "missing" / "file"); });
}

void output(const fs::path& directory) {
    const struct Field { const char* name; const char* units; double Cell::*member; } fields[] = {
        {"velocity_x", "m/s", &Cell::velocity_x}, {"velocity_y", "m/s", &Cell::velocity_y},
        {"pressure", "Pa (gauge)", &Cell::pressure}, {"density", "kg/m^3", &Cell::density},
        {"kinematic_viscosity", "m^2/s", &Cell::kinematic_viscosity},
        {"force_x", "N/m^3", &Cell::force_x}, {"force_y", "N/m^3", &Cell::force_y}
    };
    const double values[] = {-0.0, nextafter(1.0, 2.0), -3.25,
                              numeric_limits<double>::max(), numeric_limits<double>::min(),
                              numeric_limits<double>::denorm_min()};
    FluidGrid grid(3, 2, {6, 4, -3, 10});
    for (const auto& field : fields)
        for (size_t i = 0; i < 6; ++i) grid.at(i % 3, i / 3).*(field.member) = values[i];
    const auto path = directory / "grid.h5";
    save_hdf5(grid, path.string(), 1.25, 5);
    {
        H5::H5File file(path.string(), H5F_ACC_RDONLY);
        auto root = file.openGroup("/");
        check(integer_attribute(root, "format_version") == 2, "Format version.");
        check(integer_attribute(root, "width") == 3 && integer_attribute(root, "height") == 2, "Shape metadata.");
        check(integer_attribute(root, "step") == 5, "Step metadata.");
        near(real_attribute(root, "time"), 1.25);
        near(real_attribute(root, "dx"), 2); near(real_attribute(root, "dy"), 2);
        near(real_attribute(root, "length_x"), 6); near(real_attribute(root, "length_y"), 4);
        near(real_attribute(root, "origin_x"), -3); near(real_attribute(root, "origin_y"), 10);
        for (const auto& pair : {std::pair{"axis_order", "yx"}, {"field_location", "cell_center"},
                                 {"boundary_x", "periodic"}, {"boundary_y", "periodic"},
                                 {"length_units", "m"}, {"time_units", "s"}})
            check(text_attribute(root, pair.first) == pair.second, "Text metadata.");
        for (const auto& field : fields) {
            auto dataset = file.openDataSet(string("/cells/") + field.name);
            hsize_t shape[2]{};
            check(dataset.getSpace().getSimpleExtentNdims() == 2, "Dataset rank.");
            dataset.getSpace().getSimpleExtentDims(shape);
            check(shape[0] == 2 && shape[1] == 3, "Dataset axes.");
            check(dataset.getDataType() == H5::PredType::IEEE_F64LE, "Portable float64 type.");
            check(text_attribute(dataset, "units") == field.units, "Units.");
            check(dataset.getCreatePlist().getLayout() == H5D_CONTIGUOUS &&
                  dataset.getCreatePlist().getNfilters() == 0, "Storage changed unexpectedly.");
            double actual[6]{};
            dataset.read(actual, H5::PredType::NATIVE_DOUBLE);
            check(memcmp(actual, values, sizeof(values)) == 0, "HDF5 exact value round trip.");
        }
    }
    for (const auto& field : fields) {
        for (const double bad : {INFINITY, NAN}) {
            grid.at(0, 0).*(field.member) = bad;
            write_text(path, "preserve existing file");
            rejects([&] { save_hdf5(grid, path.string()); });
            check(read_text(path) == "preserve existing file", "Invalid data truncated output.");
        }
        grid.at(0, 0).*(field.member) = values[0];
    }
    rejects([&] { save_hdf5(grid, path.string(), -1); });
    rejects<runtime_error>([&] { save_hdf5(grid, (directory / "missing" / "grid.h5").string()); });
    FluidGrid single(1, 1);
    save_hdf5(single, path.string());
    const auto svg = directory / "grid.svg";
    single.at(0, 0).velocity_y = 1;
    save_svg(single, svg.string());
    const auto picture = read_text(svg);
    check(picture.find("<svg ") != string::npos && picture.find("</svg>") != string::npos,
          "SVG document missing.");
    check(picture.find("data-x=\"0\" data-y=\"0\"") != string::npos, "Missing SVG cell.");
    check(picture.find("M 280 93 L 280 75") != string::npos, "Upward SVG arrow.");
    ostringstream terminal;
    print_grid(single, terminal);
    check(terminal.str().find("0 |^|") != string::npos, "Console direction.");
    initialize_flow(single, FlowPattern::Still);
    save_svg(single, svg.string());
    check(read_text(svg).find("<circle ") != string::npos, "Still cell must be a dot.");
    single.at(0, 0).velocity_x = NAN;
    write_text(svg, "preserve");
    rejects([&] { save_svg(single, svg.string()); });
    check(read_text(svg) == "preserve", "Invalid SVG data truncated output.");
    single.at(0, 0).velocity_x = 1;
    rejects<runtime_error>([&] { save_svg(single, directory.string()); });
}

void runs(const fs::path& directory) {
    RunConfig config;
    config.width = 2; config.height = 1; config.name = "repeated-run";
    config.output_root = directory / "saves";
    const auto first = export_initial_state(config);
    const auto previous_status = read_text(first / "status.txt");
    const auto second = export_initial_state(config);
    check(first != second && first.parent_path() == config.output_root, "Run directory reuse/location.");
    check(read_text(first / "status.txt") == previous_status, "Previous run was changed.");
    check(previous_status.find("status=initialized") != string::npos, "Success status.");
    check(previous_status.find("solver=not_implemented") != string::npos, "Misleading solver status.");
    check(fs::exists(first / "grid.h5") && fs::exists(first / "grid.svg"), "Missing artifacts.");
    check(read_config(first / "resolved.cfg").width == 2, "Missing recipe.");
    const auto environment = read_text(first / "environment.txt");
    check(environment.find("hdf5_runtime=") != string::npos &&
          environment.find("source_sha256=") != string::npos, "Missing environment.");
    const auto unfinished = begin_run(config);
    check(read_text(unfinished / "status.txt").find("status=running") != string::npos, "Incomplete run label.");
    record_status(unfinished, RunStatus::Failed);
    check(read_text(unfinished / "status.txt").find("status=failed") != string::npos, "Failure label.");
    const auto next = begin_run(config);
    check(next != unfinished, "Failed run folder reused.");
    write_text(directory / "blocked", "regular file");
    config.output_root = directory / "blocked";
    rejects<fs::filesystem_error>([&] { begin_run(config); });
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "Select one test suite.");
        const string suite = argv[1];
        const auto unique = to_string(chrono::steady_clock::now().time_since_epoch().count());
        const auto directory = fs::current_path() / "test-artifacts" / (suite + "-" + unique);
        fs::create_directories(directory);
        if (suite == "geometry") geometry();
        else if (suite == "presets") presets();
        else if (suite == "simulation") simulation();
        else if (suite == "config") config(directory);
        else if (suite == "output") output(directory);
        else if (suite == "runs") runs(directory);
        else throw invalid_argument("Unknown suite.");
        cout << suite << ": passed\n";
        return 0;
    } catch (const H5::Exception& error) {
        cerr << "HDF5: " << error.getDetailMsg() << '\n';
    } catch (const exception& error) {
        cerr << error.what() << '\n';
    }
    return 1;
}
