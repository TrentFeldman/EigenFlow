// Console, HDF5 and SVG views of grid state; no simulation updates.
#include "GridOutput.h"
#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <stdexcept>

using namespace std;

namespace {
void write_text_attribute(const H5::H5Object& object, const char* name, const string& value) {
    const H5::StrType type(H5::PredType::C_S1, value.size() + 1);
    auto attribute = object.createAttribute(name, type, H5::DataSpace(H5S_SCALAR));
    attribute.write(type, value.c_str());
}

// A dominant-axis arrow keeps the console view readable without graphics.
char velocity_symbol(const Cell& cell) {
    if (!isfinite(cell.velocity_x) || !isfinite(cell.velocity_y)) {
        return '?';
    }
    if (cell.velocity_x == 0.0 && cell.velocity_y == 0.0) {
        return '.';
    }
    if (abs(cell.velocity_x) >= abs(cell.velocity_y)) {
        return cell.velocity_x > 0.0 ? '>' : '<';
    }
    return cell.velocity_y > 0.0 ? '^' : 'v';
}
} // namespace

void print_grid(const FluidGrid& grid, ostream& out) {
    const size_t label_width = to_string(grid.height() - 1).size();
    const string border = string(label_width + 1, ' ') + '+' + string(grid.width(), '-') + "+\n";
    out << "Velocity direction (+x right, +y up):\n"
        << "Legend: > right, < left, ^ up, v down, . stationary, ? invalid\n"
        << border;

    // Display rows in reverse so increasing y points up.
    for (size_t row = grid.height(); row > 0; --row) {
        const size_t y = row - 1;
        const string label = to_string(y);
        out << string(label_width - label.size(), ' ') << label << " |";
        for (size_t x = 0; x < grid.width(); ++x) {
            const Cell& cell = grid.at(x, y);
            out << velocity_symbol(cell);
        }
        out << "|\n";
    }
    out << border << "x: 0 at left, " << grid.width() - 1 << " at right\n";
}

void save_hdf5(const FluidGrid& grid, const string& path, double time, uint64_t step) {
    if (!isfinite(time) || time < 0) throw invalid_argument("Snapshot time must be finite and nonnegative.");
    struct Field {
        const char* name;
        const char* units;
        double Cell::*member;
    };
    const Field fields[] = {
        {"velocity_x", "m/s", &Cell::velocity_x},
        {"velocity_y", "m/s", &Cell::velocity_y},
        {"pressure", "Pa (gauge)", &Cell::pressure},
        {"density", "kg/m^3", &Cell::density},
        {"kinematic_viscosity", "m^2/s", &Cell::kinematic_viscosity},
        {"force_x", "N/m^3", &Cell::force_x},
        {"force_y", "N/m^3", &Cell::force_y},
    };

    // Retain validation before truncating an existing snapshot.
    for (size_t y = 0; y < grid.height(); ++y) {
        for (size_t x = 0; x < grid.width(); ++x) {
            const Cell& cell = grid.at(x, y);
            for (const Field& field : fields) {
                if (!isfinite(cell.*(field.member))) {
                    throw invalid_argument("Cannot save nonfinite cell values as HDF5.");
                }
            }
        }
    }

    // Pack one field at a time: each dataset gets one write, with one plane of scratch memory.
    vector<double> values(grid.width() * grid.height());
    H5::Exception::dontPrint();
    try {
        H5::H5File file(path, H5F_ACC_TRUNC);
        auto root = file.openGroup("/");
        const auto write_integer = [&](const char* name, uint64_t value) {
            auto attribute = root.createAttribute(name, H5::PredType::STD_U64LE,
                                                  H5::DataSpace(H5S_SCALAR));
            attribute.write(H5::PredType::NATIVE_UINT64, &value);
        };
        write_integer("format_version", 2);
        write_integer("step", step);
        write_integer("width", grid.width());
        write_integer("height", grid.height());
        write_text_attribute(root, "axis_order", "yx");
        write_text_attribute(root, "field_location", "cell_center");
        write_text_attribute(root, "boundary_x", "periodic");
        write_text_attribute(root, "boundary_y", "periodic");
        write_text_attribute(root, "length_units", "m");
        write_text_attribute(root, "time_units", "s");
        const auto write_real = [&](const char* name, double value) {
            auto attribute = root.createAttribute(name, H5::PredType::IEEE_F64LE,
                                                  H5::DataSpace(H5S_SCALAR));
            attribute.write(H5::PredType::NATIVE_DOUBLE, &value);
        };
        write_real("time", time);
        write_real("length_x", grid.domain().length_x);
        write_real("length_y", grid.domain().length_y);
        write_real("origin_x", grid.domain().origin_x);
        write_real("origin_y", grid.domain().origin_y);
        write_real("dx", grid.dx());
        write_real("dy", grid.dy());

        auto group = file.createGroup("/cells");
        const hsize_t dimensions[] = {grid.height(), grid.width()};
        const H5::DataSpace space(2, dimensions);
        for (const Field& field : fields) {
            size_t index = 0;
            for (size_t y = 0; y < grid.height(); ++y) {
                for (size_t x = 0; x < grid.width(); ++x) {
                    const Cell& cell = grid.at(x, y);
                    values[index++] = cell.*(field.member);
                }
            }
            auto dataset = group.createDataSet(field.name, H5::PredType::IEEE_F64LE, space);
            dataset.write(values.data(), H5::PredType::NATIVE_DOUBLE);
            write_text_attribute(dataset, "units", field.units);
            dataset.close();
        }
        group.close();
        root.close();
        file.flush(H5F_SCOPE_GLOBAL);
        file.close();
    } catch (const H5::Exception& error) {
        throw runtime_error("Cannot save HDF5 grid to " + path + ": " + error.getDetailMsg());
    }
}

void save_svg(const FluidGrid& grid, const string& path) {
    double max_speed = 0.0;
    for (size_t y = 0; y < grid.height(); ++y) {
        for (size_t x = 0; x < grid.width(); ++x) {
            const Cell& cell = grid.at(x, y);
            const double speed = hypot(cell.velocity_x, cell.velocity_y);
            if (!isfinite(speed)) {
                throw invalid_argument("Cannot draw nonfinite or overflowing velocity as SVG.");
            }
            max_speed = max(max_speed, speed);
        }
    }

    const double spacing = 24.0;
    const double plot_width = spacing * static_cast<double>(grid.width());
    const double plot_height = spacing * static_cast<double>(grid.height());
    const double canvas_width = max(560.0, plot_width + 112.0);
    const double canvas_height = plot_height + 136.0;
    const double left = (canvas_width - plot_width) / 2.0;
    const double top = 72.0;

    ofstream out(path);
    if (!out) throw runtime_error("Cannot open SVG output: " + path);
    out.imbue(locale::classic());
    out.precision(10);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
        << canvas_width << ' ' << canvas_height << "\" width=\"" << canvas_width
        << "\" height=\"" << canvas_height << "\">\n"
        << "<title>Fluid velocity field</title>\n"
        << "<desc>One arrow per cell. Positive x points right and positive y points up. "
        << "Arrow lengths share a scale based on this field's maximum speed.</desc>\n"
        << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n"
        << "<g font-family=\"sans-serif\" fill=\"#243247\">\n"
        << "<text x=\"24\" y=\"26\" font-size=\"18\">Velocity field: "
        << grid.width() << " x " << grid.height() << "</text>\n"
        << "<text x=\"24\" y=\"49\" font-size=\"12\">Max speed: " << max_speed
        << " m/s | length = relative speed | dots = stationary</text>\n"
        << "<defs><pattern id=\"grid\" x=\"" << left << "\" y=\"" << top
        << "\" width=\"24\" height=\"24\" patternUnits=\"userSpaceOnUse\">"
        << "<path d=\"M 24 0 H 0 V 24\" fill=\"none\" stroke=\"#e5eaf0\"/>"
        << "</pattern></defs>\n"
        << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plot_width
        << "\" height=\"" << plot_height << "\" fill=\"url(#grid)\" stroke=\"#b6c2d0\"/>\n"
        << "<g id=\"velocity\" stroke=\"#1768a6\" stroke-width=\"1.4\" "
        << "stroke-linecap=\"round\" stroke-linejoin=\"round\" fill=\"none\">\n";

    for (size_t y = 0; y < grid.height(); ++y) {
        for (size_t x = 0; x < grid.width(); ++x) {
            const Cell& cell = grid.at(x, y);
            const double cx = left + (static_cast<double>(x) + 0.5) * spacing;
            const double cy = top + (static_cast<double>(grid.height() - 1 - y) + 0.5) * spacing;
            out << "<g data-x=\"" << x << "\" data-y=\"" << y << "\"><title>Cell ("
                << x << ", " << y << "): velocity (" << cell.velocity_x << ", "
                << cell.velocity_y << ") m/s</title>";
            if (cell.velocity_x == 0.0 && cell.velocity_y == 0.0) {
                out << "<circle cx=\"" << cx << "\" cy=\"" << cy
                    << "\" r=\"1.5\" fill=\"#1768a6\" stroke=\"none\"/>";
            } else {
                // SVG y points down. A shared scale preserves relative speeds and angles.
                const double dx = (cell.velocity_x / max_speed) * spacing * 0.75;
                const double dy = -(cell.velocity_y / max_speed) * spacing * 0.75;
                const double tip_x = cx + dx / 2.0;
                const double tip_y = cy + dy / 2.0;
                out << "<path d=\"M " << cx - dx / 2.0 << ' ' << cy - dy / 2.0
                    << " L " << tip_x << ' ' << tip_y
                    << " M " << tip_x - 0.28 * dx + 0.14 * dy << ' '
                    << tip_y - 0.28 * dy - 0.14 * dx
                    << " L " << tip_x << ' ' << tip_y
                    << " L " << tip_x - 0.28 * dx - 0.14 * dy << ' '
                    << tip_y - 0.28 * dy + 0.14 * dx << "\"/>";
            }
            out << "</g>\n";
        }
    }
    out << "</g>\n<g font-size=\"12\" text-anchor=\"middle\">\n"
        << "<text x=\"" << left + spacing / 2.0 << "\" y=\"" << top + plot_height + 18.0
        << "\">0</text>\n";
    if (grid.width() > 1) {
        out << "<text x=\"" << left + plot_width - spacing / 2.0 << "\" y=\""
            << top + plot_height + 18.0 << "\">" << grid.width() - 1 << "</text>\n";
    }
    out << "<text x=\"" << canvas_width / 2.0 << "\" y=\"" << canvas_height - 14.0
        << "\">x (column); y increases upward from row 0</text>\n"
        << "</g>\n</g>\n</svg>\n";
    out.close();
    if (!out) throw runtime_error("Could not finish writing SVG output: " + path);
}
