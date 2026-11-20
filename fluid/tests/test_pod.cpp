// Regression checks for the POD toolkit (fluid_pod). Synthetic AMReX plotfiles with known analytic
// fields exercise the reader (levels, ratios, ghosts, nodal data, float32, big-endian), conversion,
// decomposition, reconstruction and export. Usage: fluid_pod_tests <suite> <work-dir>
#include "AmrexPlotfile.h"
#include "Commands.h"
#include "H5Util.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>

using namespace std;
namespace fs = filesystem;
using pod::amrex::Box;

namespace {

void check(bool condition, const string& message) {
    if (!condition) throw runtime_error(message);
}
void near_rel(double actual, double expected, double tol, const string& what) {
    check(fabs(actual - expected) <= tol * max(1.0, fabs(expected)),
          what + ": got " + to_string(actual) + ", expected " + to_string(expected));
}
int status(vector<string> args) {
    args.push_back("--quiet");
    if (args[0] == "info") args.pop_back();  // info has no --quiet
    return pod::run_cli(args);
}
void ok(const vector<string>& args) {
    string joined;
    for (auto& a : args) joined += a + " ";
    check(status(args) == 0, "command failed: fluid_pod " + joined);
}

// ---------------------------------------------------------------------------------------------
// Synthetic plotfile writer
struct SynLevel {
    Box domain;          // cell-centred
    vector<Box> boxes;   // valid boxes in cell index space; converted to the data index type on write
};
struct Syn {
    vector<string> vars = {"u", "v"};
    double time = 0;
    int step = 0;
    double plo[2] = {-1, 0}, phi[2] = {1, 1};
    vector<int> ref;
    vector<SynLevel> levels;
    int itype[2] = {0, 0};
    int nghost = 0;
    bool f32 = false, big_endian = false;
    int coord = 0;
    function<double(int, double, double, double)> f;
    double dx(int l, int d) const { return (phi[d] - plo[d]) / levels[l].domain.n(d); }
    double coord_of(int l, int d, int i) const { return plo[d] + (i + (itype[d] ? 0.0 : 0.5)) * dx(l, d); }
};

string box_str(const Box& b) {
    return "((" + to_string(b.lo[0]) + "," + to_string(b.lo[1]) + ") (" + to_string(b.hi[0]) + "," +
           to_string(b.hi[1]) + ") (" + to_string(b.type[0]) + "," + to_string(b.type[1]) + "))";
}

void write_plotfile(const fs::path& dir, const Syn& s) {
    fs::create_directories(dir);
    const int nv = (int)s.vars.size(), finest = (int)s.levels.size() - 1;
    ofstream h(dir / "Header");
    h.precision(17);
    h << "HyperCLaw-V1.1\n" << nv << "\n";
    for (auto& v : s.vars) h << v << "\n";
    h << "2\n" << s.time << "\n" << finest << "\n" << s.plo[0] << " " << s.plo[1] << " \n"
      << s.phi[0] << " " << s.phi[1] << " \n";
    for (int r : s.ref) h << r << " ";
    h << "\n";
    for (auto& L : s.levels) h << box_str(L.domain) << " ";
    h << "\n";
    for (int l = 0; l <= finest; ++l) h << s.step << " ";
    h << "\n";
    for (int l = 0; l <= finest; ++l) h << s.dx(l, 0) << " " << s.dx(l, 1) << " \n";
    h << s.coord << "\n0\n";
    for (int l = 0; l <= finest; ++l) {
        const auto& L = s.levels[l];
        h << l << " " << L.boxes.size() << " " << s.time << "\n" << s.step << "\n";
        for (auto& b : L.boxes)
            for (int d = 0; d < 2; ++d)
                h << s.plo[d] + b.lo[d] * s.dx(l, d) << " " << s.plo[d] + (b.hi[d] + 1) * s.dx(l, d) << "\n";
        h << "Level_" << l << "/Cell\n";

        fs::create_directories(dir / ("Level_" + to_string(l)));
        ofstream cellh(dir / ("Level_" + to_string(l)) / "Cell_H");
        cellh << "1\n1\n" << nv << "\n" << s.nghost << "\n(" << L.boxes.size() << " 0\n";
        vector<Box> dboxes;
        for (auto b : L.boxes) {
            for (int d = 0; d < 2; ++d) { b.hi[d] += s.itype[d]; b.type[d] = s.itype[d]; }
            dboxes.push_back(b);
            cellh << box_str(b) << "\n";
        }
        cellh << ")\n" << L.boxes.size() << "\n";
        ofstream data[2] = {ofstream(dir / ("Level_" + to_string(l)) / "Cell_D_00000", ios::binary),
                            ofstream(dir / ("Level_" + to_string(l)) / "Cell_D_00001", ios::binary)};
        long offset[2] = {0, 0};
        for (size_t k = 0; k < dboxes.size(); ++k) {
            const int fi = int(k % 2);
            cellh << "FabOnDisk: Cell_D_0000" << fi << " " << offset[fi] << "\n";
            Box fb = dboxes[k];
            for (int d = 0; d < 2; ++d) { fb.lo[d] -= s.nghost; fb.hi[d] += s.nghost; }
            string hdr = s.f32 ? "FAB ((4, (32 8 23 0 1 9 0 127)),(4, (4 3 2 1)))"
                               : s.big_endian ? "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (1 2 3 4 5 6 7 8)))"
                                              : "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))";
            hdr += box_str(fb) + " " + to_string(nv) + "\n";
            string bytes;
            for (int c = 0; c < nv; ++c)
                for (int j = fb.lo[1]; j <= fb.hi[1]; ++j)
                    for (int i = fb.lo[0]; i <= fb.hi[0]; ++i) {
                        const bool ghost = i < dboxes[k].lo[0] || i > dboxes[k].hi[0] || j < dboxes[k].lo[1] ||
                                           j > dboxes[k].hi[1];
                        const double v = ghost ? 1e30 : s.f(c, s.coord_of(l, 0, i), s.coord_of(l, 1, j), s.time);
                        char b8[8];
                        if (s.f32) { float x = (float)v; memcpy(b8, &x, 4); bytes.append(b8, 4); continue; }
                        memcpy(b8, &v, 8);
                        if (s.big_endian) reverse(b8, b8 + 8);
                        bytes.append(b8, 8);
                    }
            data[fi] << hdr << bytes;
            offset[fi] += (long)(hdr.size() + bytes.size());
        }
    }
}

Box box(int x0, int y0, int x1, int y1) {
    Box b;
    b.lo[0] = x0; b.lo[1] = y0; b.hi[0] = x1; b.hi[1] = y1;
    return b;
}

double field(int c, double x, double y, double t) { return (c + 1) * (1.5 * x - 0.7 * y) + 0.25 * t + c; }

// Three levels (ratios 2 and 4) covering nested patches; level 0 split into two grids.
Syn three_level() {
    Syn s;
    s.f = field;
    s.ref = {2, 4};
    s.levels.push_back({box(0, 0, 7, 3), {box(0, 0, 3, 3), box(4, 0, 7, 3)}});
    s.levels.push_back({box(0, 0, 15, 7), {box(4, 2, 11, 5)}});
    s.levels.push_back({box(0, 0, 63, 31), {box(24, 12, 39, 19), box(40, 12, 43, 15)}});
    return s;
}

// Expected composite value at target index (i, j) given the level that supplied it.
double expected(const Syn& s, int c, int i, int j, int level) {
    int r = 1;
    for (int k = level; k + 1 < (int)s.levels.size(); ++k) r *= s.ref[k];
    double v = s.f(c, s.coord_of(level, 0, i / r), s.coord_of(level, 1, j / r), s.time);
    return s.f32 ? (double)(float)v : v;
}

void verify_composite(const Syn& s, const fs::path& dir, const string& what) {
    auto pf = pod::amrex::read_header(dir);
    auto g = pod::amrex::make_grid({pf});
    vector<double> out;
    vector<unsigned char> mask;
    pod::amrex::composite(pf, {0, 1}, g, out, mask);
    const int last = (int)s.levels.size() - 1;
    check(g.nx() == s.levels[last].domain.n(0) + s.itype[0] && g.ny() == s.levels[last].domain.n(1) + s.itype[1],
          what + ": grid size");
    vector<int> per_level(s.levels.size(), 0);
    for (int j = 0; j < g.ny(); ++j)
        for (int i = 0; i < g.nx(); ++i) {
            const size_t p = (size_t)j * g.nx() + i;
            check(mask[p] < s.levels.size(), what + ": bad mask");
            per_level[mask[p]]++;
            for (int c = 0; c < 2; ++c)
                check(out[c * g.npt() + p] == expected(s, c, i, j, mask[p]),
                      what + ": value mismatch at (" + to_string(i) + "," + to_string(j) + ")");
        }
    for (size_t l = 0; l < s.levels.size(); ++l) check(per_level[l] > 0, what + ": level unused");
}

// ---------------------------------------------------------------------------------------------
void reader(const fs::path& work) {
    Syn s = three_level();
    s.time = 0.5;
    s.step = 7;
    s.nghost = 1;
    write_plotfile(work / "plt00007", s);
    auto pf = pod::amrex::read_header(work / "plt00007");
    check(pf.vars == s.vars && pf.finest == 2 && pf.step() == 7 && pf.time == 0.5, "header fields");
    check(pf.ref == vector<int>{2, 4} && pf.levels[2].boxes.size() == 2 && pf.levels[1].nghost == 1, "levels");
    check(pod::amrex::series_of(work / "plt00007") == "plt", "series name");
    verify_composite(s, work / "plt00007", "cell-centred, 3 levels, ghosts");

    Syn n = three_level();
    n.itype[0] = n.itype[1] = 1;
    write_plotfile(work / "nodal00000", n);
    verify_composite(n, work / "nodal00000", "nodal");

    Syn f = three_level();
    f.f32 = true;
    f.nghost = 2;
    write_plotfile(work / "single00000", f);
    verify_composite(f, work / "single00000", "float32");

    Syn b = three_level();
    b.big_endian = true;
    write_plotfile(work / "big00000", b);
    verify_composite(b, work / "big00000", "big-endian");

    // A 3D header must be rejected with a clear message.
    fs::create_directories(work / "plt3d00000");
    ofstream(work / "plt3d00000" / "Header") << "HyperCLaw-V1.1\n1\nu\n3\n0\n0\n";
    try {
        pod::amrex::read_header(work / "plt3d00000");
        check(false, "3D header accepted");
    } catch (const pod::Error& e) {
        check(string(e.what()).find("2D only") != string::npos, "3D error message");
    }
    // A truncated FAB must be reported, not read as garbage.
    fs::resize_file(work / "big00000" / "Level_2" / "Cell_D_00001", 100);
    vector<double> out;
    vector<unsigned char> mask;
    auto pf2 = pod::amrex::read_header(work / "big00000");
    bool threw = false;
    try { pod::amrex::composite(pf2, {0}, pod::amrex::make_grid({pf2}), out, mask); }
    catch (const pod::Error&) { threw = true; }
    check(threw, "truncated FAB not detected");
}

// Run directory: a series whose finest level appears after the first snapshot, a second series,
// a checkpoint and an AMReX backup directory.
fs::path make_run(const fs::path& run) {
    for (int k = 0; k < 4; ++k) {
        Syn s = three_level();
        s.step = 10 * k;
        s.time = 0.1 * k;
        if (k == 0) { s.levels.resize(1); s.ref.clear(); }  // only the coarse level at step 0
        write_plotfile(run / ("plt" + string(5 - to_string(s.step).size(), '0') + to_string(s.step)), s);
        Syn o = three_level();
        o.vars = {"a", "b"};
        o.levels.resize(2);
        o.ref = {2};
        o.step = 10 * k;
        o.time = 0.1 * k;
        write_plotfile(run / ("other" + to_string(s.step)), o);
    }
    fs::create_directories(run / "chk00010");
    ofstream(run / "chk00010" / "Header") << "CheckPointVersion_1.0\n";
    Syn backup = three_level();
    backup.step = 99;
    write_plotfile(run / "plt00010.old.123456", backup);
    return run;
}

void convert(const fs::path& work) {
    const fs::path run = make_run(work / "run");
    const string out = (work / "run.h5").string();
    ok({"convert", run.string(), "-o", out, "-n"});
    check(!fs::exists(out), "dry run wrote output");
    ok({"convert", run.string(), "-o", out, "-j", "3"});
    check(fs::exists(work / "run.plt.vtkhdf") && fs::exists(work / "run.other.vtkhdf"), "vtkhdf sidecars");
    check(!fs::exists(out + ".partial"), "partial file left behind");
    check(status({"convert", run.string(), "-o", out}) == 1, "overwrote without --force");
    {
        pod::h5::Handle f = pod::h5::open_file(out);
        check(pod::h5::read_attr_str(f, "format") == "eigenflow-snapshots", "format attribute");
        auto step = pod::h5::read<long long>(f, "plt/step");
        check(step == vector<long long>{0, 10, 20, 30}, "steps sorted, backup skipped");
        check(pod::h5::dims(f, "plt/fields/u") == vector<hsize_t>{4, 32, 64}, "plt shape on finest grid");
        check(pod::h5::dims(f, "other/fields/a") == vector<hsize_t>{4, 8, 16}, "other shape");
        auto mask = pod::h5::read<unsigned char>(f, "plt/level_mask");
        for (size_t p = 0; p < 32 * 64; ++p) check(mask[p] == 0, "step 0 must come from level 0 only");
        auto u = pod::h5::read<double>(f, "plt/fields/u"), x = pod::h5::read<double>(f, "plt/x");
        Syn s = three_level();
        s.time = 0.1 * 3;  // same arithmetic as make_run (0.1*3 != 0.3 in binary)
        for (int j = 0; j < 32; ++j)
            for (int i = 0; i < 64; ++i) {
                const size_t p = 3 * 32 * 64 + (size_t)j * 64 + i;
                check(u[p] == expected(s, 0, i, j, mask[p]), "converted value");
            }
        near_rel(x[0], -1 + 1.0 / 64, 1e-15, "x centre");
    }
    const string sub = (work / "sub.h5").string();
    ok({"convert", run.string(), "-o", sub, "-s", "plt", "-v", "v", "--steps", "10:20", "--f32", "--no-mask", "--gzip", "0"});
    {
        pod::h5::Handle f = pod::h5::open_file(sub);
        check(!pod::h5::exists(f, "other") && !pod::h5::exists(f, "plt/fields/u") && !pod::h5::exists(f, "plt/level_mask"),
              "filters");
        check(pod::h5::read<long long>(f, "plt/step") == vector<long long>{10, 20}, "--steps");
    }
    check(status({"convert", run.string(), "-o", sub, "-f", "-v", "nope*"}) == 1, "unmatched variable accepted");
    ok({"convert", run.string(), "-o", sub, "-f", "--every", "2", "-s", "other"});
    {
        pod::h5::Handle f = pod::h5::open_file(sub);
        check(pod::h5::read<long long>(f, "other/step") == vector<long long>{0, 20}, "--every");
    }
    ok({"info", run.string()});
    ok({"info", out});
}

// Analytic rank-2 fluctuation: u = m(x,y) + sin(t) p1(x,y) + cos(2t) p2(x,y), likewise v.
double rank2(int c, double x, double y, double t) {
    const double m = 1 + 0.3 * x * y, p1 = sin(3.0 * x) * cos(2.0 * y), p2 = x * x - y;
    return (c ? -0.5 : 1.0) * m + sin(t) * p1 * (c ? 2.0 : 1.0) + cos(2 * t) * p2 * (c ? -1.0 : 0.5);
}

fs::path make_rank2(const fs::path& work, int coord) {
    const fs::path dir = work / (coord ? "rz" : "xy");
    for (int k = 0; k < 12; ++k) {
        Syn s;
        s.f = rank2;
        s.coord = coord;
        if (coord) { s.plo[0] = 0; s.phi[0] = 2; }
        s.levels.push_back({box(0, 0, 31, 15), {box(0, 0, 31, 15)}});
        s.step = k;
        s.time = 0.37 * k;
        write_plotfile(dir / ("plt" + to_string(1000 + k)), s);
    }
    const string h5 = (work / (coord ? "rz.h5" : "xy.h5")).string();
    ok({"convert", dir.string(), "-o", h5});
    return h5;
}

void pod_suite(const fs::path& work) {
    for (int coord : {0, 1}) {
        const string h5 = make_rank2(work, coord), tag = coord ? "rz" : "xy";
        const string svd = (work / (tag + "_svd.h5")).string(), gram = (work / (tag + "_gram.h5")).string();
        ok({"pod", h5, "-v", "u,v", "-k", "4", "-o", svd, "--check"});
        ok({"pod", h5, "-v", "*", "-k", "4", "-o", gram, "-m", "gram"});
        pod::h5::Handle fs_ = pod::h5::open_file(svd), fg = pod::h5::open_file(gram);
        check(pod::h5::read_attr<int>(fs_, "numerical_rank")[0] == 2, tag + ": rank-2 data must have rank 2");
        check(pod::h5::read_attr<int>(fg, "numerical_rank")[0] == 2, tag + ": gram rank");
        check(pod::h5::read_attr_str(fs_, "weighting") == (coord ? "rz" : "area"), tag + ": weighting");
        auto ec = pod::h5::read<double>(fs_, "cumulative_energy");
        near_rel(ec[1], 1.0, 1e-12, tag + ": two modes capture all energy");
        auto ss = pod::h5::read<double>(fs_, "singular_values"), sg = pod::h5::read<double>(fg, "singular_values");
        for (int i = 0; i < 2; ++i) near_rel(sg[i], ss[i], 1e-9, tag + ": svd vs gram singular values");
        // Weighted orthonormality of stored modes: sum over vars and cells of w * phi_i * phi_j.
        auto w = pod::h5::read<double>(fs_, "weights");
        auto mu = pod::h5::read<double>(fs_, "modes/u"), mv = pod::h5::read<double>(fs_, "modes/v");
        const size_t npt = w.size();
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                double d = 0;
                for (size_t p = 0; p < npt; ++p)
                    d += w[p] * (mu[i * npt + p] * mu[j * npt + p] + mv[i * npt + p] * mv[j * npt + p]);
                near_rel(d, i == j ? 1.0 : 0.0, 1e-10, tag + ": modes orthonormal");
            }
    }
    const string h5 = (work / "xy.h5").string();
    const string e = (work / "e.h5").string(), r = (work / "r.h5").string();
    ok({"pod", h5, "-v", "u", "-e", "0.5", "-o", e});
    {
        pod::h5::Handle f = pod::h5::open_file(e);
        check(pod::h5::dims(f, "coeffs")[1] == 1, "--energy picks the fewest modes");
    }
    ok({"pod", h5, "-v", "u", "--region", "0:0.5,0.25:", "-t", "2:10:2", "--normalize", "-o", r, "--csv",
        (work / "r").string()});
    {
        pod::h5::Handle f = pod::h5::open_file(r);
        auto x = pod::h5::read<double>(f, "x"), y = pod::h5::read<double>(f, "y");
        check(x.front() >= 0 && x.back() <= 0.5 && y.front() >= 0.25, "--region bounds");
        check(pod::h5::read<int>(f, "snapshot_index") == vector<int>{2, 4, 6, 8}, "--snapshots slice");
    }
    check(fs::exists(work / "r_energy.csv") && fs::exists(work / "r_coeffs.csv"), "csv output");
    check(status({"pod", h5, "-o", (work / "x.h5").string()}) == 1, "missing -v accepted");
    check(status({"pod", h5, "-v", "u", "-m", "qr", "-o", (work / "x.h5").string()}) == 1, "bad method accepted");
}

void reconstruct(const fs::path& work) {
    const string h5 = make_rank2(work, 0);
    const string p = (work / "p.h5").string(), rec = (work / "rec.h5").string();
    ok({"pod", h5, "-v", "u,v", "-k", "3", "-o", p});
    ok({"reconstruct", p, "-k", "2", "-o", rec, "--compare-csv", (work / "err.csv").string()});
    pod::h5::Handle fo = pod::h5::open_file(h5), fr = pod::h5::open_file(rec);
    for (string v : {"u", "v"}) {
        auto a = pod::h5::read<double>(fo, "plt/fields/" + v), b = pod::h5::read<double>(fr, "plt_rank2/fields/" + v);
        check(a.size() == b.size(), "reconstruction shape");
        double err = 0, nrm = 0;
        for (size_t i = 0; i < a.size(); ++i) { err = max(err, fabs(a[i] - b[i])); nrm = max(nrm, fabs(a[i])); }
        check(err <= 1e-10 * nrm, "rank-2 reconstruction must be exact for " + v);
    }
    check(fs::exists(work / "rec.vtkhdf"), "reconstruction sidecar");
    // Rank 1 must be worse, and the reconstruction must be readable by pod again (same layout).
    ok({"reconstruct", p, "-k", "1", "-o", (work / "rec1.h5").string(), "--compare"});
    ok({"pod", rec, "-v", "u", "-o", (work / "again.h5").string()});
    check(status({"reconstruct", p, "-k", "9"}) == 1, "requesting more modes than stored accepted");
}

void export_suite(const fs::path& work) {
    const string h5 = make_rank2(work, 0);
    const fs::path dir = work / "npy";
    ok({"export", h5, "-d", dir.string()});
    ifstream in(dir / "plt.fields.u.npy", ios::binary);
    string magic(6, '\0');
    in.read(&magic[0], 6);
    check(magic == "\x93NUMPY", "npy magic");
    in.seekg(8);
    unsigned char lb[2];
    in.read((char*)lb, 2);
    const int hl = lb[0] | (lb[1] << 8);
    string hdr(hl, '\0');
    in.read(&hdr[0], hl);
    check((10 + hl) % 64 == 0, "npy header alignment");
    check(hdr.find("'descr': '<f8'") != string::npos && hdr.find("(12, 16, 32)") != string::npos, "npy header: " + hdr);
    check(fs::file_size(dir / "plt.fields.u.npy") == (uintmax_t)(10 + hl + 12 * 16 * 32 * 8), "npy size");
    check(fs::exists(dir / "plt.level_mask.npy") && fs::exists(dir / "attributes.json"), "export completeness");
    check(status({"export", h5, "-d", dir.string()}) == 1, "export into existing dir without -f");
    ok({"export", h5, "--list"});
}

void cli(const fs::path&) {
    check(pod::run_cli({"--help"}) == 0, "--help");
    check(pod::run_cli({"pod", "--help"}) == 0, "pod --help");
    check(pod::run_cli({"--version"}) == 0, "--version");
    check(pod::run_cli({}) == 1, "no command");
    check(pod::run_cli({"frobnicate"}) == 1, "unknown command");
    check(pod::run_cli({"convert", "--bogus"}) == 1, "unknown option");
    check(pod::run_cli({"convert"}) == 1, "missing input");
    check(pod::run_cli({"pod", "/nonexistent.h5", "-v", "u"}) == 1, "missing file");
    check(pod::glob_match("x_vel*", "x_velocity") && !pod::glob_match("x_vel?", "x_velocity"), "glob");
    auto s = pod::parse_slice("1:-1:3", 10);
    check(s.start == 1 && s.step == 3 && s.count == 3, "slice");
    check(pod::parse_slice("-1", 10).start == 9, "negative index");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "usage: fluid_pod_tests <reader|convert|pod|reconstruct|export|cli> <work-dir>\n";
        return 2;
    }
    const string suite = argv[1];
    const fs::path work = fs::path(argv[2]) / suite;
    fs::remove_all(work);
    fs::create_directories(work);
    const map<string, function<void(const fs::path&)>> suites = {
        {"reader", reader}, {"convert", convert}, {"pod", pod_suite},
        {"reconstruct", reconstruct}, {"export", export_suite}, {"cli", cli}};
    auto it = suites.find(suite);
    if (it == suites.end()) { cerr << "unknown suite " << suite << "\n"; return 2; }
    try {
        it->second(work);
    } catch (const exception& e) {
        cerr << suite << ": FAILED: " << e.what() << "\n";
        return 1;
    }
    cout << suite << ": ok\n";
    return 0;
}
