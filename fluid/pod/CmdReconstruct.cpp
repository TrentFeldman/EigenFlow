// fluid_pod reconstruct: rebuild snapshots from the first K POD modes, optionally comparing to the source.
#include "Commands.h"
#include "SnapshotFile.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace pod {

namespace fs = std::filesystem;

int cmd_reconstruct(const std::vector<std::string>& argv) {
    const std::vector<Opt> opts = {
        {"modes", 'k', "K", "number of modes to use (default: all stored)"},
        {"energy", 'e', "FRAC", "use the fewest stored modes that capture this energy fraction"},
        {"output", 'o', "FILE", "output snapshot file (default: <pod-stem>.rank<K>.h5)"},
        {"compare", 'c', "", "report relative L2 error against the original snapshots"},
        {"compare-csv", 0, "FILE", "write per-snapshot errors to FILE (implies --compare)"},
        {"no-output", 0, "", "only compare; do not write reconstructed fields"},
        {"f32", 0, "", "store reconstructed fields as float32"},
        {"no-vtk", 0, "", "do not write a ParaView .vtkhdf sidecar"},
        {"force", 'f', "", "overwrite an existing output file"},
        {"quiet", 'q', "", "only print errors"},
    };
    auto a = parse_args(argv, "fluid_pod reconstruct [options] POD.h5",
                        "Rebuild the decomposed snapshots from the first K modes:\n"
                        "  field(t) = mean + var_scale * sum_{i<K} coeffs[t,i] * modes[i].\n"
                        "The output uses the same layout as 'convert', so every tool here can read it.",
                        opts);
    if (a.has("quiet")) verbosity() = 0;
    if (a.pos.size() != 1) fail("expected exactly one POD file (see --help)");
    const std::string in = a.pos[0];
    const bool compare = a.has("compare") || a.has("compare-csv");
    if (a.has("no-output") && !compare) fail("--no-output without --compare does nothing");
    Timer timer;

    h5::Handle f = h5::open_file(in);
    if (h5::read_attr_str(f, "format", "") != kPodFormat) fail(in, " is not a POD file from 'fluid_pod pod'");
    const auto vars = split(h5::read_attr_str(f, "variables"), ',');
    const std::string series = h5::read_attr_str(f, "series");
    const auto scale = h5::read_attr<double>(f, "var_scale");
    const auto region = h5::read_attr<int>(f, "region_index");
    const auto x = h5::read<double>(f, "x"), y = h5::read<double>(f, "y"), time = h5::read<double>(f, "time");
    const auto snap = h5::read<int>(f, "snapshot_index");
    const auto ec = h5::read<double>(f, "cumulative_energy");
    const auto cd = h5::dims(f, "coeffs");
    const int n = (int)cd[0], kstored = (int)cd[1], nx = (int)x.size(), ny = (int)y.size(), nv = (int)vars.size();
    const size_t npr = (size_t)nx * ny;
    const auto coeffs = h5::read<double>(f, "coeffs");
    const auto w = h5::read<double>(f, "weights");
    auto lo = h5::read_attr<double>(f, "prob_lo"), dxv = h5::read_attr<double>(f, "dx");
    auto itv = h5::read_attr<int>(f, "index_type");
    const double prob_lo[2] = {lo[0], lo[1]}, dx[2] = {dxv[0], dxv[1]};
    const int itype[2] = {itv[0], itv[1]};
    const int coord_sys = h5::read_attr<int>(f, "coord_sys").at(0);

    int k = (int)a.get_long("modes", kstored);
    if (a.has("energy")) {
        const double target = a.get_double("energy", 0.99);
        int ke = kstored;
        for (int i = 0; i < kstored; ++i)
            if (ec[i] >= target - 1e-15) { ke = i + 1; break; }
        k = a.has("modes") ? std::min(k, ke) : ke;
    }
    if (k < 0) fail("--modes must be >= 0");
    if (k > kstored) fail("only ", kstored, " modes are stored; rerun 'fluid_pod pod' with -k ", k);

    // rec[c][t][p]
    std::vector<std::vector<double>> rec(nv, std::vector<double>((size_t)n * npr));
    for (int c = 0; c < nv; ++c) {
        const auto mean = h5::read<double>(f, "mean/" + vars[c]);
        const auto modes = h5::read<double>(f, "modes/" + vars[c]);
        for (int t = 0; t < n; ++t) {
            double* dst = rec[c].data() + (size_t)t * npr;
            std::fill(dst, dst + npr, 0.0);
            for (int i = 0; i < k; ++i) {
                const double ai = coeffs[(size_t)t * kstored + i];
                const double* phi = modes.data() + (size_t)i * npr;
                for (size_t p = 0; p < npr; ++p) dst[p] += ai * phi[p];
            }
            for (size_t p = 0; p < npr; ++p) dst[p] = mean[p] + scale[c] * dst[p];
        }
    }

    if (compare) {
        const std::string src = h5::read_attr_str(f, "source");
        if (!fs::exists(src)) fail("source file ", src, " not found (needed for --compare)");
        h5::Handle fs_ = h5::open_file(src);
        h5::Handle gf = h5::open_group(fs_, series + "/fields");
        std::vector<std::vector<double>> err(nv, std::vector<double>(n));
        double tot_e = 0, tot_n = 0;
        for (int c = 0; c < nv; ++c) {
            for (int t = 0; t < n; ++t) {
                auto orig = h5::read_slab<double>(gf, vars[c], {(hsize_t)snap[t], (hsize_t)region[2], (hsize_t)region[0]},
                                                  {1, 1, 1}, {1, (hsize_t)ny, (hsize_t)nx});
                double e2 = 0, n2 = 0;
                const double* r = rec[c].data() + (size_t)t * npr;
                for (size_t p = 0; p < npr; ++p) {
                    e2 += w[p] * (r[p] - orig[p]) * (r[p] - orig[p]);
                    n2 += w[p] * orig[p] * orig[p];
                }
                err[c][t] = n2 > 0 ? std::sqrt(e2 / n2) : std::sqrt(e2);
                tot_e += e2;
                tot_n += n2;
            }
        }
        std::printf("rank-%d reconstruction vs %s:/%s (weighted relative L2 of the full field)\n", k, src.c_str(),
                    series.c_str());
        std::printf("  %-20s %12s %12s\n", "variable", "mean err", "max err");
        for (int c = 0; c < nv; ++c) {
            double mean_e = 0, max_e = 0;
            for (double e : err[c]) { mean_e += e / n; max_e = std::max(max_e, e); }
            std::printf("  %-20s %12.4e %12.4e\n", vars[c].c_str(), mean_e, max_e);
        }
        std::printf("  %-20s %12.4e\n", "all (pooled)", tot_n > 0 ? std::sqrt(tot_e / tot_n) : 0.0);
        if (a.has("compare-csv")) {
            std::ofstream o(a.get("compare-csv"));
            if (!o) fail("cannot write ", a.get("compare-csv"));
            o << "snapshot,time";
            for (auto& v : vars) o << "," << v;
            o << "\n";
            o.precision(10);
            for (int t = 0; t < n; ++t) {
                o << snap[t] << "," << time[t];
                for (int c = 0; c < nv; ++c) o << "," << err[c][t];
                o << "\n";
            }
        }
    }

    if (a.has("no-output")) return 0;
    fs::path out = a.has("output") ? fs::path(a.get("output"))
                                   : fs::path(fs::path(in).stem().string() + ".rank" + std::to_string(k) + ".h5");
    if (fs::exists(out) && !a.has("force")) fail(out.string(), " already exists (use -f to overwrite)");
    const std::string gname = series + "_rank" + std::to_string(k);
    {
        h5::Handle fo = h5::create_file(out.string());
        h5::attr_str(fo, "format", kSnapshotFormat);
        h5::attr(fo, "format_version", 1);
        h5::attr_str(fo, "created_by", cat("fluid_pod ", version(), " reconstruct"));
        h5::attr_str(fo, "inputs", fs::absolute(in).lexically_normal().string());
        h5::Handle g = h5::create_group(fo, gname);
        h5::attr_str(g, "variables", join(vars));
        h5::attr(g, "rank", k);
        write_geometry_attrs(g, prob_lo, dx, coord_sys, itype, x, y);
        h5::write(g, "x", x);
        h5::write(g, "y", y);
        h5::write(g, "time", time);
        std::vector<long long> step(snap.begin(), snap.end());
        h5::write(g, "step", step);
        h5::write(g, "snapshot_index", snap);
        h5::Handle gf = h5::create_group(g, "fields");
        for (int c = 0; c < nv; ++c)
            h5::write(gf, vars[c], rec[c], {(hsize_t)n, (hsize_t)ny, (hsize_t)nx}, 4,
                      a.has("f32") ? H5T_IEEE_F32LE : H5T_IEEE_F64LE);
    }
    if (!a.has("no-vtk")) {
        std::vector<VtkField> fields;
        for (auto& v : vars) fields.push_back({v, "/" + gname + "/fields/" + v, {}});
        fs::path sc = out;
        sc.replace_extension(".vtkhdf");
        write_vtkhdf(sc, out, time, vtk_grid(x, y, dx, itype), fields);
    }
    if (verbosity() >= 1) std::printf("wrote %s (series /%s, %d snapshots, rank %d) in %.2f s\n", out.string().c_str(), gname.c_str(), n, k,
                timer.seconds());
    return 0;
}

}  // namespace pod
