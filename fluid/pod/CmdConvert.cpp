// fluid_pod convert: AMReX plotfile series -> one HDF5 file (+ ParaView .vtkhdf sidecars).
#include "AmrexPlotfile.h"
#include "Commands.h"
#include "SnapshotFile.h"

#include <cstdio>
#include <filesystem>

namespace pod {

namespace fs = std::filesystem;

namespace {

struct SeriesPlan {
    std::string name;
    std::vector<amrex::Plotfile> pfs;
    amrex::Grid grid;
    std::vector<std::string> vars;
    std::vector<int> comps;
};

fs::path default_output(const std::string& first_input) {
    fs::path p = fs::path(first_input).lexically_normal();
    if (p.filename().empty()) p = p.parent_path();
    if (amrex::is_plotfile(p)) p = fs::absolute(p).parent_path();
    std::string stem = fs::absolute(p).filename().string();
    return fs::path((stem.empty() ? "snapshots" : stem) + ".h5");
}

}  // namespace

int cmd_convert(const std::vector<std::string>& argv) {
    const std::vector<Opt> opts = {
        {"output", 'o', "FILE", "output HDF5 file (default: <run-dir-name>.h5 in the current directory)"},
        {"series", 's', "GLOBS", "only these plotfile series, e.g. 'nddataPLT' or 'MOF*,nd*'"},
        {"vars", 'v', "GLOBS", "only these variables, e.g. 'x_velocity,y_velocity,PRES*'"},
        {"steps", 0, "A:B", "only plotfiles whose step number is in [A, B] (either side optional)"},
        {"times", 0, "A:B", "only plotfiles whose time is in [A, B]"},
        {"every", 0, "N", "keep every Nth plotfile after the filters above"},
        {"f32", 0, "", "store fields as float32 (half the size; POD still computes in float64)"},
        {"gzip", 0, "LEVEL", "gzip level 0-9 (default 4; 0 = uncompressed, fastest)"},
        {"no-mask", 0, "", "do not store level_mask"},
        {"no-vtk", 0, "", "do not write ParaView .vtkhdf sidecar files"},
        {"jobs", 'j', "N", "reader threads (default: all cores)"},
        {"force", 'f', "", "overwrite an existing output file"},
        {"dry-run", 'n', "", "show what would be converted, write nothing"},
        {"quiet", 'q', "", "only print errors"},
        {"verbose", 0, "", "print progress per batch and skipped directories"},
    };
    auto a = parse_args(argv, "fluid_pod convert [options] PATH...",
                        "Composite 2D AMReX plotfiles onto a uniform grid and store them as one HDF5 file.\n"
                        "PATH is a run directory (scanned one level deep) or individual plotfile directories.\n"
                        "Plotfiles are grouped into series by name with the step digits removed\n"
                        "(nddataPLT00000030 -> /nddataPLT). Checkpoints (chk*) and *.old.* backups are skipped.",
                        opts);
    if (a.has("quiet")) verbosity() = 0;
    if (a.has("verbose")) verbosity() = 2;
    if (a.pos.empty()) fail("no input paths (give a run directory or plotfile directories; see --help)");
    const int jobs = (int)a.get_long("jobs", default_threads());
    const int gzip = (int)a.get_long("gzip", 4);
    if (gzip < 0 || gzip > 9) fail("--gzip must be 0-9");
    const long every = a.get_long("every", 1);
    if (every < 1) fail("--every must be >= 1");
    const Range steps = parse_range(a.get("steps")), times = parse_range(a.get("times"));
    const bool f32 = a.has("f32"), mask_on = !a.has("no-mask");
    Timer timer;

    // ---- discover and plan ----
    auto disc = amrex::discover(a.pos);
    for (auto& s : disc.skipped) debug("skip " + s.first + ": " + s.second);
    std::vector<std::string> names;
    for (auto& kv : disc.series) names.push_back(kv.first);
    if (names.empty()) fail("no AMReX plotfiles found in: ", join(a.pos, " "));
    names = select_names(names, a.get("series"), "series");

    std::vector<SeriesPlan> plans;
    std::set<std::string> matched_patterns;
    const auto var_patterns = split(a.get("vars"), ',');
    for (auto& name : names) {
        const auto& dirs = disc.series[name];
        SeriesPlan p;
        p.name = name;
        p.pfs.resize(dirs.size());
        parallel_for(dirs.size(), jobs, [&](size_t i) { p.pfs[i] = amrex::read_header(dirs[i]); });
        std::stable_sort(p.pfs.begin(), p.pfs.end(), [](const amrex::Plotfile& x, const amrex::Plotfile& y) {
            return x.step() != y.step() ? x.step() < y.step() : x.time < y.time;
        });
        std::vector<amrex::Plotfile> kept;
        for (auto& pf : p.pfs)
            if (steps.contains(pf.step()) && times.contains(pf.time)) kept.push_back(std::move(pf));
        p.pfs.clear();
        for (size_t i = 0; i < kept.size(); i += every) p.pfs.push_back(std::move(kept[i]));
        if (p.pfs.empty()) { warn("series " + name + ": no plotfiles left after --steps/--times filters"); continue; }
        for (auto& pat : var_patterns)
            for (auto& v : p.pfs[0].vars)
                if (glob_match(pat, v)) matched_patterns.insert(pat);
        p.vars = select_names(p.pfs[0].vars, a.get("vars"), "variable", true);
        if (p.vars.empty()) { info("series " + name + ": no matching variables, skipped"); continue; }
        for (auto& v : p.vars)
            p.comps.push_back(int(std::find(p.pfs[0].vars.begin(), p.pfs[0].vars.end(), v) - p.pfs[0].vars.begin()));
        p.grid = amrex::make_grid(p.pfs);
        plans.push_back(std::move(p));
    }
    for (auto& pat : var_patterns)
        if (!matched_patterns.count(pat)) fail("variable pattern '", pat, "' matches nothing in the selected series");
    if (plans.empty()) fail("nothing to convert");

    // ---- plan summary ----
    double total_bytes = 0;
    const bool show = verbosity() >= 1 || a.has("dry-run");
    if (show) std::printf("%-14s %6s %13s %15s %9s %7s %6s %12s\n", "series", "files", "steps", "time", "grid", "levels",
                "vars", "raw size");
    for (auto& p : plans) {
        const double bytes = (double)p.pfs.size() * p.grid.npt() * p.vars.size() * (f32 ? 4 : 8);
        total_bytes += bytes;
        int maxlev = 0;
        for (auto& pf : p.pfs) maxlev = std::max(maxlev, pf.finest + 1);
        if (show) std::printf("%-14s %6zu %6d..%-6d %7.4g..%-7.4g %4dx%-4d %7d %6zu %12s\n", p.name.c_str(), p.pfs.size(),
                    p.pfs.front().step(), p.pfs.back().step(), p.pfs.front().time, p.pfs.back().time, p.grid.nx(),
                    p.grid.ny(), maxlev, p.vars.size(), human_bytes(bytes).c_str());
    }
    if (!disc.skipped.empty()) info(cat("skipped ", disc.skipped.size(), " non-plotfile entries (--verbose lists them)"));
    if (a.has("dry-run")) {
        std::printf("dry run: would write %s (uncompressed %s)\n",
                    (a.has("output") ? fs::path(a.get("output")) : default_output(a.pos[0])).string().c_str(),
                    human_bytes(total_bytes).c_str());
        return 0;
    }

    const fs::path out = a.has("output") ? fs::path(a.get("output")) : default_output(a.pos[0]);
    if (fs::exists(out) && !a.has("force")) fail(out.string(), " already exists (use -f to overwrite)");
    if (!out.parent_path().empty()) fs::create_directories(out.parent_path());
    const fs::path partial = out.string() + ".partial";
    std::vector<fs::path> sidecars;
    struct RemoveOnFailure {  // never leave a half-written file behind
        fs::path p;
        bool done = false;
        ~RemoveOnFailure() { if (!done) { std::error_code ec; fs::remove(p, ec); } }
    } guard{partial};

    {
        h5::Handle f = h5::create_file(partial.string());
        h5::attr_str(f, "format", kSnapshotFormat);
        h5::attr(f, "format_version", 1);
        h5::attr_str(f, "created_by", cat("fluid_pod ", version()));
        std::vector<std::string> inputs;
        for (auto& p : a.pos) inputs.push_back(fs::absolute(p).lexically_normal().string());
        h5::attr_str(f, "inputs", join(inputs, "\n"));

        for (auto& p : plans) {
            const auto& G = p.grid;
            const hsize_t nt = p.pfs.size(), Ny = G.ny(), Nx = G.nx();
            const size_t npt = G.npt(), nv = p.vars.size();
            std::vector<double> x(Nx), y(Ny), time(nt);
            std::vector<long long> step(nt);
            std::vector<std::string> pnames;
            for (int i = 0; i < (int)Nx; ++i) x[i] = G.coord(0, i);
            for (int j = 0; j < (int)Ny; ++j) y[j] = G.coord(1, j);
            for (size_t t = 0; t < nt; ++t) {
                time[t] = p.pfs[t].time;
                step[t] = p.pfs[t].step();
                pnames.push_back(p.pfs[t].name);
            }

            h5::Handle g = h5::create_group(f, p.name);
            h5::attr_str(g, "variables", join(p.vars));
            h5::attr_str(g, "all_plotfile_variables", join(p.pfs[0].vars));
            write_geometry_attrs(g, G.prob_lo, G.dx, G.coord_sys, G.itype, x, y);
            h5::attr(g, "prob_hi", std::vector<double>{G.prob_hi[0], G.prob_hi[1]});
            h5::attr(g, "target_level", G.target_level);
            if (!G.ref.empty()) h5::attr(g, "ref_ratio", G.ref);
            h5::attr_str(g, "plotfiles", join(pnames, "\n"));
            h5::write(g, "x", x);
            h5::write(g, "y", y);
            h5::write(g, "time", time);
            h5::write(g, "step", step);

            h5::Handle gf = h5::create_group(g, "fields");
            std::vector<h5::Handle> dsets;
            for (auto& v : p.vars)
                dsets.push_back(h5::create_dataset(gf, v, f32 ? H5T_IEEE_F32LE : H5T_IEEE_F64LE, {nt, Ny, Nx}, gzip));
            h5::Handle dmask;
            if (mask_on) dmask = h5::create_dataset(g, "level_mask", H5T_STD_U8LE, {nt, Ny, Nx}, gzip);

            // Bounded-memory batches: parallel composite, serial HDF5 writes.
            const double snap_bytes = (double)npt * (nv * 8 + 1);
            const size_t batch =
                std::max<size_t>(1, std::min<size_t>((size_t)jobs * 2, (size_t)(1024.0 * 1024 * 1024 / snap_bytes)));
            std::vector<std::vector<double>> buf(batch);
            std::vector<std::vector<unsigned char>> mbuf(batch);
            for (size_t t0 = 0; t0 < nt; t0 += batch) {
                const size_t nb = std::min<size_t>(batch, nt - t0);
                parallel_for(nb, jobs, [&](size_t k) { amrex::composite(p.pfs[t0 + k], p.comps, G, buf[k], mbuf[k]); });
                for (size_t k = 0; k < nb; ++k) {
                    for (size_t c = 0; c < nv; ++c)
                        h5::write_slab(dsets[c], {t0 + k, 0, 0}, {1, Ny, Nx}, buf[k].data() + c * npt);
                    if (mask_on) h5::write_slab(dmask, {t0 + k, 0, 0}, {1, Ny, Nx}, mbuf[k].data());
                }
                debug(cat("  ", p.name, ": ", t0 + nb, "/", nt));
            }

            if (!a.has("no-vtk")) {
                std::vector<VtkField> fields;
                for (auto& v : p.vars) fields.push_back({v, "/" + p.name + "/fields/" + v, {}});
                if (mask_on) fields.push_back({"level_mask", "/" + p.name + "/level_mask", {}});
                fs::path sc = out.parent_path() / (out.stem().string() + "." + p.name + ".vtkhdf");
                write_vtkhdf(sc, out, time, vtk_grid(x, y, G.dx, G.itype), fields);
                sidecars.push_back(sc);
            }
        }
    }
    fs::rename(partial, out);
    guard.done = true;
    if (verbosity() >= 1) {
        std::printf("wrote %s (%s) in %.2f s\n", out.string().c_str(), human_bytes((double)fs::file_size(out)).c_str(),
                    timer.seconds());
        for (auto& sc : sidecars) std::printf("      %s  (open in ParaView)\n", sc.string().c_str());
    }
    return 0;
}

}  // namespace pod
