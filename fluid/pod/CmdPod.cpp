// fluid_pod pod: proper orthogonal decomposition of a snapshot series.
//
// X (m x n): column t = stacked selected variables of snapshot t over the selected region.
// Optionally subtract the temporal mean and scale each variable by its fluctuation RMS, then
// with cell weights W (area, or 2*pi*r*area for RZ) decompose  W^1/2 X = U S V^T.
//   modes   phi_i = W^-1/2 U[:, i]     orthonormal in the weighted inner product: phi^T W phi = I
//   coeffs  a_i(t) = s_i V[t, i]
//   energy  s_i^2 / sum_j s_j^2
// Reconstruction: field(t) = mean + scale * sum_i a_i(t) phi_i.
#include "Commands.h"
#include "Lapack.h"
#include "SnapshotFile.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace pod {

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Region {
    int i0, i1, j0, j1;  // inclusive
};

Region parse_region(const std::string& s, const SeriesMeta& m) {
    Region r{0, m.nx() - 1, 0, m.ny() - 1};
    if (s.empty()) return r;
    auto parts = split(s, ',');
    if (parts.empty() || parts.size() > 2) fail("--region expects X0:X1,Y0:Y1 (either side may be empty)");
    auto apply = [&](const std::string& spec, const std::vector<double>& c, int& lo, int& hi, const char* axis) {
        Range rg = parse_range(spec);
        lo = -1;
        hi = -1;
        for (int i = 0; i < (int)c.size(); ++i)
            if (rg.contains(c[i])) {
                if (lo < 0) lo = i;
                hi = i;
            }
        if (lo < 0) fail("--region: no ", axis, " coordinates inside '", spec, "' (domain ", c.front(), "..", c.back(), ")");
    };
    // allow ",Y0:Y1" by writing an empty x part
    if (s[0] == ',') {
        apply(parts[0], m.y, r.j0, r.j1, "y");
        return r;
    }
    if (!parts[0].empty() && parts[0] != ":") apply(parts[0], m.x, r.i0, r.i1, "x");
    if (parts.size() == 2 && parts[1] != ":") apply(parts[1], m.y, r.j0, r.j1, "y");
    return r;
}

void write_csv_energy(const fs::path& p, const std::vector<double>& s, const std::vector<double>& ef,
                      const std::vector<double>& ec) {
    std::ofstream o(p);
    if (!o) fail("cannot write ", p.string());
    o << "mode,singular_value,energy_fraction,cumulative_energy\n";
    o.precision(17);
    for (size_t i = 0; i < s.size(); ++i) o << i + 1 << "," << s[i] << "," << ef[i] << "," << ec[i] << "\n";
}

void write_csv_coeffs(const fs::path& p, const std::vector<double>& time, const std::vector<int>& idx,
                      const std::vector<double>& a, int k) {
    std::ofstream o(p);
    if (!o) fail("cannot write ", p.string());
    o << "snapshot,time";
    for (int i = 0; i < k; ++i) o << ",a" << i + 1;
    o << "\n";
    o.precision(17);
    for (size_t t = 0; t < time.size(); ++t) {
        o << idx[t] << "," << time[t];
        for (int i = 0; i < k; ++i) o << "," << a[t * k + i];
        o << "\n";
    }
}

}  // namespace

int cmd_pod(const std::vector<std::string>& argv) {
    const std::vector<Opt> opts = {
        {"series", 's', "NAME", "series to decompose (optional if the file has only one)"},
        {"vars", 'v', "GLOBS", "variables to stack into the state vector (required), e.g. 'x_velocity,y_velocity'"},
        {"modes", 'k', "K", "number of modes to store (default 10; capped at the rank)"},
        {"energy", 'e', "FRAC", "store the fewest modes that capture this energy fraction, e.g. 0.99"},
        {"snapshots", 't', "SLICE", "snapshot indices, Python slice start:stop:step (default all), e.g. 5: or ::2"},
        {"region", 'r', "X0:X1,Y0:Y1", "restrict to a physical sub-rectangle, e.g. '-0.5:1,0:0.5' or ',0.2:'"},
        {"no-mean", 0, "", "do not subtract the temporal mean (decompose raw snapshots)"},
        {"normalize", 0, "", "scale each variable by its fluctuation RMS (use when mixing units)"},
        {"no-weights", 0, "", "plain Euclidean inner product instead of cell-area (RZ: 2*pi*r) weights"},
        {"method", 'm', "svd|gram", "svd: thin SVD (accurate, default); gram: method of snapshots (less memory, faster for many points)"},
        {"check", 0, "", "verify orthonormality and reconstruction error after decomposing"},
        {"output", 'o', "FILE", "output file (default: <input-stem>.<series>.pod.h5)"},
        {"csv", 0, "PREFIX", "also write PREFIX_energy.csv and PREFIX_coeffs.csv"},
        {"no-vtk", 0, "", "do not write a ParaView .vtkhdf sidecar for the modes"},
        {"force", 'f', "", "overwrite an existing output file"},
        {"quiet", 'q', "", "do not print the energy table"},
    };
    auto a = parse_args(argv, "fluid_pod pod [options] SNAPSHOTS.h5",
                        "Proper orthogonal decomposition of one snapshot series produced by 'fluid_pod convert'.", opts);
    if (a.has("quiet")) verbosity() = 0;
    if (a.pos.size() != 1) fail("expected exactly one input file (see --help)");
    const std::string in = a.pos[0];
    const std::string method = a.get("method", "svd");
    if (method != "svd" && method != "gram") fail("--method must be svd or gram");
    const bool subtract_mean = !a.has("no-mean"), normalize = a.has("normalize"), weighted = !a.has("no-weights");
    Timer timer;

    h5::Handle f = h5::open_file(in);
    const std::string series = pick_series(f, a.get("series"), in);
    const SeriesMeta meta = read_series(f, series);
    if (!a.has("vars"))
        fail("choose variables with -v (globs allowed). ", series, " has: ", join(meta.vars, ", "));
    const auto vars = select_names(meta.vars, a.get("vars"), "variable");
    const Slice sl = parse_slice(a.get("snapshots"), meta.nt());
    const Region rg = parse_region(a.get("region"), meta);
    const int nx = rg.i1 - rg.i0 + 1, ny = rg.j1 - rg.j0 + 1, nv = (int)vars.size(), n = sl.count;
    const size_t npr = (size_t)nx * ny;
    const size_t m_sz = npr * nv;
    if (m_sz > 2147483647ull) fail("state vector too large for 32-bit LAPACK (", m_sz, " rows); use --region");
    const int m = (int)m_sz;
    if (n < 2 && subtract_mean) fail("need at least 2 snapshots when subtracting the mean");
    info(cat("POD of /", series, " [", join(vars), "]: ", n, " snapshots x ", m, " dofs (", nx, "x", ny, " grid, ",
             human_bytes(8.0 * m * n), " matrix), method ", method));

    // ---- snapshot matrix, column-major ----
    std::vector<double> X((size_t)m * n);
    std::vector<int> snap_idx(n);
    std::vector<double> time(n);
    for (int t = 0; t < n; ++t) {
        snap_idx[t] = sl.at(t);
        time[t] = meta.time[snap_idx[t]];
    }
    {
        h5::Handle gf = h5::open_group(f, series + "/fields");
        for (int c = 0; c < nv; ++c) {
            auto v = h5::read_slab<double>(gf, vars[c], {(hsize_t)sl.start, (hsize_t)rg.j0, (hsize_t)rg.i0},
                                           {(hsize_t)sl.step, 1, 1}, {(hsize_t)n, (hsize_t)ny, (hsize_t)nx});
            for (int t = 0; t < n; ++t)
                std::copy_n(v.data() + (size_t)t * npr, npr, X.data() + (size_t)t * m + c * npr);
        }
    }
    for (double v : X)
        if (!std::isfinite(v)) fail("snapshot matrix contains NaN/Inf; restrict --vars/--region/--snapshots");

    // ---- weights ----
    std::vector<double> w(npr, 1.0);
    std::string weighting = "none";
    if (weighted) {
        const double area = meta.dx[0] * meta.dx[1];
        weighting = "area";
        if (meta.coord_sys == 1) weighting = "rz";
        else if (meta.coord_sys == 2) warn("spherical coordinates: using plain cell-area weights");
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                w[(size_t)j * nx + i] =
                    meta.coord_sys == 1 ? 2 * kPi * std::fabs(meta.x[rg.i0 + i]) * area : area;
    }
    double wsum = 0;
    for (double v : w) wsum += v;
    if (!(wsum > 0)) fail("all weights are zero (RZ grid with r = 0 everywhere?)");

    // ---- mean, scaling, weighting ----
    std::vector<double> mean(m, 0.0), scale(nv, 1.0);
    if (subtract_mean) {
        for (int t = 0; t < n; ++t)
            for (int r = 0; r < m; ++r) mean[r] += X[(size_t)t * m + r];
        for (auto& v : mean) v /= n;
        for (int t = 0; t < n; ++t)
            for (int r = 0; r < m; ++r) X[(size_t)t * m + r] -= mean[r];
    }
    if (normalize)
        for (int c = 0; c < nv; ++c) {
            double ss = 0;
            for (int t = 0; t < n; ++t)
                for (size_t p = 0; p < npr; ++p) {
                    const double v = X[(size_t)t * m + c * npr + p];
                    ss += w[p] * v * v;
                }
            const double rms = std::sqrt(ss / (wsum * n));
            scale[c] = rms > 0 ? rms : 1.0;
            for (int t = 0; t < n; ++t)
                for (size_t p = 0; p < npr; ++p) X[(size_t)t * m + c * npr + p] /= scale[c];
        }
    std::vector<double> sqw(npr);
    for (size_t p = 0; p < npr; ++p) sqw[p] = std::sqrt(w[p]);
    for (int t = 0; t < n; ++t)
        for (int c = 0; c < nv; ++c)
            for (size_t p = 0; p < npr; ++p) X[(size_t)t * m + c * npr + p] *= sqw[p];

    // ---- decomposition ----
    const int r = std::min(m, n);
    std::vector<double> s(r), U, VT((size_t)r * n);  // VT: r x n column-major (ld r)
    std::vector<double> Xcopy;
    if (a.has("check") && method == "svd") Xcopy = X;
    int info_ = 0;
    if (method == "svd") {
        U.resize((size_t)m * r);
        std::vector<int> iwork(8 * (size_t)r);
        int lwork = -1;
        double wq = 0;
        dgesdd_("S", &m, &n, X.data(), &m, s.data(), U.data(), &m, VT.data(), &r, &wq, &lwork, iwork.data(), &info_, 1);
        lwork = (int)wq;
        std::vector<double> work(std::max(1, lwork));
        dgesdd_("S", &m, &n, X.data(), &m, s.data(), U.data(), &m, VT.data(), &r, work.data(), &lwork, iwork.data(),
                &info_, 1);
        if (info_ != 0) fail("LAPACK dgesdd failed (info=", info_, ")");
        X.swap(Xcopy);  // X now holds the weighted matrix only if --check
    } else {
        // Method of snapshots: K = X^T X (n x n), K V = V diag(lambda), U = X V / s.
        std::vector<double> K((size_t)n * n, 0.0), lam(n);
        const double one = 1.0, zero = 0.0;
        dsyrk_("U", "T", &n, &m, &one, X.data(), &m, &zero, K.data(), &n, 1, 1);
        int lwork = -1, liwork = -1, iwq = 0;
        double wq = 0;
        dsyevd_("V", "U", &n, K.data(), &n, lam.data(), &wq, &lwork, &iwq, &liwork, &info_, 1, 1);
        lwork = (int)wq;
        liwork = iwq;
        std::vector<double> work(std::max(1, lwork));
        std::vector<int> iwork(std::max(1, liwork));
        dsyevd_("V", "U", &n, K.data(), &n, lam.data(), work.data(), &lwork, iwork.data(), &liwork, &info_, 1, 1);
        if (info_ != 0) fail("LAPACK dsyevd failed (info=", info_, ")");
        // eigenvalues ascending -> descending
        std::vector<double> V((size_t)n * r);
        for (int i = 0; i < r; ++i) {
            const int src = n - 1 - i;
            s[i] = std::sqrt(std::max(0.0, lam[src]));
            std::copy_n(K.data() + (size_t)src * n, n, V.data() + (size_t)i * n);
            for (int t = 0; t < n; ++t) VT[(size_t)t * r + i] = V[(size_t)i * n + t];
        }
        U.assign((size_t)m * r, 0.0);
        dgemm_("N", "N", &m, &r, &n, &one, X.data(), &m, V.data(), &n, &zero, U.data(), &m, 1, 1);
        const double tiny = (s.empty() ? 0 : s[0]) * 1e-10;
        for (int i = 0; i < r; ++i) {
            if (s[i] <= tiny) {
                std::fill_n(U.data() + (size_t)i * m, m, 0.0);
                continue;
            }
            for (int row = 0; row < m; ++row) U[(size_t)i * m + row] /= s[i];
        }
    }

    // ---- energy and mode count ----
    double tot = 0;
    for (double v : s) tot += v * v;
    std::vector<double> ef(r), ec(r);
    for (int i = 0; i < r; ++i) {
        ef[i] = tot > 0 ? s[i] * s[i] / tot : 0;
        ec[i] = (i ? ec[i - 1] : 0) + ef[i];
    }
    int k = (int)a.get_long("modes", 10);
    if (a.has("energy")) {
        const double target = a.get_double("energy", 0.99);
        if (!(target > 0 && target <= 1)) fail("--energy must be in (0, 1]");
        int ke = r;
        for (int i = 0; i < r; ++i)
            if (ec[i] >= target - 1e-15) { ke = i + 1; break; }
        k = a.has("modes") ? std::min(k, ke) : ke;
    }
    if (k < 1) fail("--modes must be >= 1");
    if (k > r) k = r;
    // Singular values below this are numerically zero. The Gram route squares the condition number,
    // so its floor is ~sqrt(eps) instead of ~eps.
    const double zero_tol =
        (s.empty() ? 0 : s[0]) * (method == "gram" ? 1e-7 : std::max(m, n) * 2.2e-16);
    int numerical_rank = 0;
    for (double v : s) numerical_rank += v > zero_tol;

    // ---- optional checks (in the weighted space) ----
    if (a.has("check")) {
        double ortho = 0;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j <= i; ++j) {
                if (s[i] <= zero_tol || s[j] <= zero_tol) continue;
                double d = 0;
                for (int row = 0; row < m; ++row) d += U[(size_t)i * m + row] * U[(size_t)j * m + row];
                ortho = std::max(ortho, std::fabs(d - (i == j ? 1.0 : 0.0)));
            }
        double err2 = 0, nrm2 = 0;
        for (int t = 0; t < n; ++t)
            for (int row = 0; row < m; ++row) {
                double acc = 0;
                for (int i = 0; i < k; ++i) acc += U[(size_t)i * m + row] * s[i] * VT[(size_t)t * r + i];
                const double ref = X[(size_t)t * m + row];
                err2 += (acc - ref) * (acc - ref);
                nrm2 += ref * ref;
            }
        const double actual = nrm2 > 0 ? std::sqrt(err2 / nrm2) : 0;
        const double predicted = std::sqrt(std::max(0.0, 1 - ec[k - 1]));
        std::printf("check: max|phi^T W phi - I| = %.2e; rank-%d error %.6e (predicted from singular values %.6e)\n",
                    ortho, k, actual, predicted);
    }

    // ---- unweight modes, coefficients ----
    std::vector<double> coeffs((size_t)n * k);
    for (int t = 0; t < n; ++t)
        for (int i = 0; i < k; ++i) coeffs[(size_t)t * k + i] = s[i] * VT[(size_t)t * r + i];
    for (int i = 0; i < k; ++i)
        for (int c = 0; c < nv; ++c)
            for (size_t p = 0; p < npr; ++p) U[(size_t)i * m + c * npr + p] /= sqw[p] > 0 ? sqw[p] : 1.0;

    // ---- output ----
    fs::path out = a.has("output") ? fs::path(a.get("output"))
                                   : fs::path(fs::path(in).stem().string() + "." + series + ".pod.h5");
    if (fs::exists(out) && !a.has("force")) fail(out.string(), " already exists (use -f to overwrite)");
    std::vector<double> xs(meta.x.begin() + rg.i0, meta.x.begin() + rg.i1 + 1);
    std::vector<double> ys(meta.y.begin() + rg.j0, meta.y.begin() + rg.j1 + 1);
    {
        h5::Handle fo = h5::create_file(out.string());
        h5::attr_str(fo, "format", kPodFormat);
        h5::attr(fo, "format_version", 1);
        h5::attr_str(fo, "created_by", cat("fluid_pod ", version()));
        h5::attr_str(fo, "source", fs::absolute(in).lexically_normal().string());
        h5::attr_str(fo, "series", series);
        h5::attr_str(fo, "variables", join(vars));
        h5::attr(fo, "mean_subtracted", (int)subtract_mean);
        h5::attr(fo, "normalized", (int)normalize);
        h5::attr_str(fo, "weighting", weighting);
        h5::attr_str(fo, "method", method);
        h5::attr(fo, "var_scale", scale);
        h5::attr(fo, "region_index", std::vector<int>{rg.i0, rg.i1, rg.j0, rg.j1});
        h5::attr(fo, "numerical_rank", numerical_rank);
        write_geometry_attrs(fo, meta.prob_lo, meta.dx, meta.coord_sys, meta.itype, xs, ys);
        h5::write(fo, "x", xs);
        h5::write(fo, "y", ys);
        h5::write(fo, "time", time);
        h5::write(fo, "snapshot_index", snap_idx);
        h5::write(fo, "singular_values", s);
        h5::write(fo, "energy_fraction", ef);
        h5::write(fo, "cumulative_energy", ec);
        h5::write(fo, "coeffs", coeffs, {(hsize_t)n, (hsize_t)k});
        h5::write(fo, "weights", w, {(hsize_t)ny, (hsize_t)nx});
        h5::Handle gm = h5::create_group(fo, "mean"), gmode = h5::create_group(fo, "modes");
        std::vector<double> buf((size_t)k * npr);
        for (int c = 0; c < nv; ++c) {
            h5::write(gm, vars[c], std::vector<double>(mean.begin() + c * npr, mean.begin() + (c + 1) * npr),
                      {(hsize_t)ny, (hsize_t)nx});
            for (int i = 0; i < k; ++i)
                std::copy_n(U.data() + (size_t)i * m + c * npr, npr, buf.data() + (size_t)i * npr);
            h5::write(gmode, vars[c], buf, {(hsize_t)k, (hsize_t)ny, (hsize_t)nx}, 4);
        }
    }
    if (!a.has("no-vtk")) {  // ParaView "time" = mode number
        std::vector<VtkField> fields;
        for (auto& v : vars) fields.push_back({"mode_" + v, "/modes/" + v, {}});
        for (int c = 0; c < nv; ++c)
            fields.push_back({"mean_" + vars[c], "", std::vector<double>(mean.begin() + c * npr, mean.begin() + (c + 1) * npr)});
        std::vector<double> mode_no(k);
        for (int i = 0; i < k; ++i) mode_no[i] = i + 1;
        fs::path sc = out;
        sc.replace_extension(".vtkhdf");
        write_vtkhdf(sc, out, mode_no, vtk_grid(xs, ys, meta.dx, meta.itype), fields);
    }
    if (a.has("csv")) {
        const std::string pre = a.get("csv");
        write_csv_energy(pre + "_energy.csv", s, ef, ec);
        write_csv_coeffs(pre + "_coeffs.csv", time, snap_idx, coeffs, k);
    }

    if (verbosity() >= 1) {
        std::printf("%5s %14s %10s %10s %12s\n", "mode", "sigma", "energy%", "cum%", "rel.err");
        const int rows = std::min(r, std::max(k, 10));
        for (int i = 0; i < rows; ++i)
            std::printf("%5d %14.6e %10.4f %10.4f %12.4e%s\n", i + 1, s[i], 100 * ef[i], 100 * ec[i],
                        std::sqrt(std::max(0.0, 1 - ec[i])), i + 1 == k ? "   <- stored" : "");
        if (rows < r) std::printf("  ... %d more (all singular values are in the output file)\n", r - rows);
    }
    if (verbosity() >= 1) std::printf("wrote %s: %d modes of %d (numerical rank %d), %.4f%% energy, %.2f s\n", out.string().c_str(), k, r,
                numerical_rank, 100 * ec[k - 1], timer.seconds());
    return 0;
}

}  // namespace pod
