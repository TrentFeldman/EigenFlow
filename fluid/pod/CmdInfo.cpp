// fluid_pod info: describe run directories, plotfiles, snapshot files and POD files.
#include "AmrexPlotfile.h"
#include "Commands.h"
#include "SnapshotFile.h"

#include <cmath>
#include <cstdio>
#include <filesystem>

namespace pod {

namespace fs = std::filesystem;

namespace {

void print_wrapped(const std::vector<std::string>& items, const std::string& indent) {
    std::string line = indent;
    for (size_t i = 0; i < items.size(); ++i) {
        std::string item = items[i] + (i + 1 < items.size() ? ", " : "");
        if (line.size() + item.size() > 100 && line.size() > indent.size()) {
            std::printf("%s\n", line.c_str());
            line = indent;
        }
        line += item;
    }
    if (line.size() > indent.size()) std::printf("%s\n", line.c_str());
}

void info_plotfile(const fs::path& p) {
    auto pf = amrex::read_header(p);
    std::printf("%s: AMReX plotfile, step %d, time %.10g\n", p.string().c_str(), pf.step(), pf.time);
    std::printf("  domain [%g, %g] x [%g, %g], %s coordinates, %s-centred data\n", pf.prob_lo[0], pf.prob_hi[0],
                pf.prob_lo[1], pf.prob_hi[1], amrex::coord_name(pf.coord_sys), amrex::centering_name(pf.itype).c_str());
    for (int l = 0; l <= pf.finest; ++l) {
        const auto& L = pf.levels[l];
        long long cells = 0;
        for (auto& b : L.boxes) cells += (long long)b.n(0) * b.n(1);
        const double cover = 100.0 * cells / ((double)L.domain.n(0) * L.domain.n(1));
        std::printf("  level %d: %dx%d domain, dx %g x %g, %zu grids, %.1f%% covered, %d ghost%s\n", l,
                    L.domain.n(0), L.domain.n(1), L.dx[0], L.dx[1], L.boxes.size(), cover, L.nghost,
                    l < pf.finest ? cat(", ref ratio ", pf.ref[l]).c_str() : "");
    }
    std::printf("  %zu variables:\n", pf.vars.size());
    print_wrapped(pf.vars, "    ");
}

void info_run(const std::vector<std::string>& paths, bool show_vars) {
    auto disc = amrex::discover(paths);
    if (disc.series.empty()) fail("no AMReX plotfiles found in: ", join(paths, " "));
    std::printf("%-14s %6s %15s %17s %9s %7s %5s\n", "series", "files", "steps", "time", "grid", "levels", "vars");
    for (auto& kv : disc.series) {
        std::vector<amrex::Plotfile> pfs;
        for (auto& d : kv.second) pfs.push_back(amrex::read_header(d));
        std::sort(pfs.begin(), pfs.end(), [](auto& x, auto& y) { return x.step() < y.step(); });
        auto g = amrex::make_grid(pfs);
        int lmin = 99, lmax = 0;
        for (auto& p : pfs) { lmin = std::min(lmin, p.finest + 1); lmax = std::max(lmax, p.finest + 1); }
        std::printf("%-14s %6zu %7d..%-7d %8.4g..%-8.4g %4dx%-4d %7s %5zu\n", kv.first.c_str(), pfs.size(),
                    pfs.front().step(), pfs.back().step(), pfs.front().time, pfs.back().time, g.nx(), g.ny(),
                    (lmin == lmax ? std::to_string(lmax) : cat(lmin, "-", lmax)).c_str(), pfs[0].vars.size());
        if (show_vars) print_wrapped(pfs[0].vars, "    ");
    }
    std::map<std::string, std::vector<std::string>> by_reason;
    for (auto& s : disc.skipped) by_reason[s.second].push_back(fs::path(s.first).filename().string());
    for (auto& kv : by_reason)
        std::printf("skipped %zu: %s (%s%s)\n", kv.second.size(), kv.first.c_str(), kv.second.front().c_str(),
                    kv.second.size() > 1 ? (" .. " + kv.second.back()).c_str() : "");
    if (!show_vars) std::printf("(--vars lists variable names)\n");
}

void info_h5(const std::string& path, bool show_vars) {
    h5::Handle f = h5::open_file(path);
    const std::string fmt = h5::read_attr_str(f, "format", "");
    std::printf("%s: %s (%s)\n", path.c_str(), fmt.empty() ? "HDF5 file" : fmt.c_str(),
                human_bytes((double)fs::file_size(path)).c_str());
    if (fmt == kSnapshotFormat) {
        for (auto& s : list_series(f)) {
            auto m = read_series(f, s);
            std::printf("  /%s: %d snapshots, t = %.6g..%.6g, %dx%d %s grid, x [%g, %g], y [%g, %g], %zu variables\n",
                        s.c_str(), m.nt(), m.time.empty() ? 0 : m.time.front(), m.time.empty() ? 0 : m.time.back(),
                        m.nx(), m.ny(), m.centering.c_str(), m.x.front(), m.x.back(), m.y.front(), m.y.back(),
                        m.vars.size());
            if (show_vars) print_wrapped(m.vars, "    ");
        }
        if (!show_vars) std::printf("(--vars lists variable names)\n");
    } else if (fmt == kPodFormat) {
        const auto s = h5::read<double>(f, "singular_values");
        const auto ef = h5::read<double>(f, "energy_fraction"), ec = h5::read<double>(f, "cumulative_energy");
        const auto cd = h5::dims(f, "coeffs");
        const auto x = h5::read<double>(f, "x"), y = h5::read<double>(f, "y");
        std::printf("  source %s:/%s\n  variables %s\n", h5::read_attr_str(f, "source").c_str(),
                    h5::read_attr_str(f, "series").c_str(), h5::read_attr_str(f, "variables").c_str());
        std::printf("  %llu snapshots, %llu modes stored, %dx%d grid (x [%g, %g], y [%g, %g])\n",
                    (unsigned long long)cd[0], (unsigned long long)cd[1], (int)x.size(), (int)y.size(), x.front(),
                    x.back(), y.front(), y.back());
        std::printf("  method %s, weighting %s, mean %s, normalized %s\n", h5::read_attr_str(f, "method").c_str(),
                    h5::read_attr_str(f, "weighting").c_str(),
                    h5::read_attr<int>(f, "mean_subtracted")[0] ? "removed" : "kept",
                    h5::read_attr<int>(f, "normalized")[0] ? "yes" : "no");
        std::printf("  %5s %14s %10s %10s\n", "mode", "sigma", "energy%", "cum%");
        for (size_t i = 0; i < std::min<size_t>(s.size(), std::max<size_t>(cd[1], 10)); ++i)
            std::printf("  %5zu %14.6e %10.4f %10.4f\n", i + 1, s[i], 100 * ef[i], 100 * ec[i]);
    } else {
        std::printf("  (not written by fluid_pod; 'fluid_pod export --list %s' lists its datasets)\n", path.c_str());
    }
}

}  // namespace

int cmd_info(const std::vector<std::string>& argv) {
    const std::vector<Opt> opts = {
        {"vars", 'v', "", "list variable names"},
    };
    auto a = parse_args(argv, "fluid_pod info [options] PATH...",
                        "Describe a run directory (series, steps, times, grids), a single plotfile (levels,\n"
                        "grids, coverage, variables), or an HDF5 file written by fluid_pod.",
                        opts);
    if (a.pos.empty()) fail("expected a path (see --help)");
    std::vector<std::string> dirs;
    for (auto& p : a.pos) {
        if (fs::is_regular_file(p)) {
            if (!dirs.empty()) { info_run(dirs, a.has("vars")); dirs.clear(); }
            info_h5(p, a.has("vars"));
        } else if (amrex::is_plotfile(p) && a.pos.size() == 1) {
            info_plotfile(p);
        } else {
            dirs.push_back(p);
        }
    }
    if (!dirs.empty()) info_run(dirs, a.has("vars"));
    return 0;
}

}  // namespace pod
