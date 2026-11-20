#include "SnapshotFile.h"

#include "AmrexPlotfile.h"

namespace pod {

std::vector<std::string> list_series(hid_t file) {
    std::vector<std::string> out;
    for (auto& c : h5::children(file))
        if (h5::is_group(file, c) && h5::is_group(file, c + "/fields")) out.push_back(c);
    return out;
}

SeriesMeta read_series(hid_t file, const std::string& name) {
    SeriesMeta m;
    m.name = name;
    h5::Handle g = h5::open_group(file, name);
    m.vars = split(h5::read_attr_str(g, "variables"), ',');
    m.x = h5::read<double>(g, "x");
    m.y = h5::read<double>(g, "y");
    m.time = h5::read<double>(g, "time");
    m.step = h5::exists(g, "step") ? h5::read<long long>(g, "step") : std::vector<long long>(m.time.size());
    auto lo = h5::read_attr<double>(g, "prob_lo"), dx = h5::read_attr<double>(g, "dx");
    auto it = h5::has_attr(g, "index_type") ? h5::read_attr<int>(g, "index_type") : std::vector<int>{0, 0};
    for (int d = 0; d < 2; ++d) {
        m.prob_lo[d] = lo.at(d);
        m.dx[d] = dx.at(d);
        m.itype[d] = it.at(d);
    }
    m.coord_sys = h5::has_attr(g, "coord_sys") ? h5::read_attr<int>(g, "coord_sys").at(0) : 0;
    m.centering = amrex::centering_name(m.itype);
    return m;
}

std::string pick_series(hid_t file, const std::string& requested, const std::string& path) {
    auto all = list_series(file);
    if (all.empty()) fail(path, ": no snapshot series found (expected output of 'fluid_pod convert')");
    if (requested.empty()) {
        if (all.size() == 1) return all[0];
        fail(path, " has several series; choose one with -s: ", join(all, ", "));
    }
    auto hit = select_names(all, requested, "series");
    if (hit.size() != 1) fail("-s '", requested, "' matches several series: ", join(hit, ", "));
    return hit[0];
}

void write_geometry_attrs(hid_t obj, const double prob_lo[2], const double dx[2], int coord_sys, const int itype[2],
                          const std::vector<double>& x, const std::vector<double>& y) {
    h5::attr(obj, "prob_lo", std::vector<double>{prob_lo[0], prob_lo[1]});
    h5::attr(obj, "dx", std::vector<double>{dx[0], dx[1]});
    h5::attr(obj, "origin", std::vector<double>{x.empty() ? 0 : x[0], y.empty() ? 0 : y[0]});
    h5::attr(obj, "coord_sys", coord_sys);
    h5::attr_str(obj, "coord_sys_name", amrex::coord_name(coord_sys));
    h5::attr(obj, "index_type", std::vector<int>{itype[0], itype[1]});
    h5::attr_str(obj, "centering", amrex::centering_name(itype));
    h5::attr_str(obj, "axis_order", "t,y,x");
}

VtkGrid vtk_grid(const std::vector<double>& x, const std::vector<double>& y, const double dx[2],
                   const int itype[2]) {
    VtkGrid g;
    g.nx = (int)x.size();
    g.ny = (int)y.size();
    g.spacing[0] = dx[0];
    g.spacing[1] = dx[1];
    g.node_centered = itype[0] && itype[1];
    // Face-centred data is shown on cells (offset by half a cell); exact staggering is in the attributes.
    g.origin[0] = x.empty() ? 0 : x[0] - (g.node_centered ? 0 : 0.5 * dx[0]);
    g.origin[1] = y.empty() ? 0 : y[0] - (g.node_centered ? 0 : 0.5 * dx[1]);
    return g;
}

}  // namespace pod
