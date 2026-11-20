#include "AmrexPlotfile.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>

namespace pod::amrex {

namespace {

std::vector<std::string> read_lines(const fs::path& p) {
    std::ifstream f(p);
    if (!f) fail("cannot open ", p.string());
    std::vector<std::string> L;
    for (std::string s; std::getline(f, s);) {
        if (!s.empty() && s.back() == '\r') s.pop_back();
        L.push_back(s);
    }
    return L;
}

std::vector<Box> parse_boxes(const std::string& s) {
    static const std::regex re(R"(\(\((-?\d+),(-?\d+)\)\s*\((-?\d+),(-?\d+)\)\s*\((\d+),(\d+)\)\))");
    std::vector<Box> out;
    for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it) {
        Box b;
        for (int d = 0; d < 2; ++d) {
            b.lo[d] = std::stoi((*it)[1 + d]);
            b.hi[d] = std::stoi((*it)[3 + d]);
            b.type[d] = std::stoi((*it)[5 + d]);
        }
        out.push_back(b);
    }
    return out;
}

template <class T> std::vector<T> parse_nums(const std::string& s) {
    std::istringstream is(s);
    std::vector<T> v;
    for (T x; is >> x;) v.push_back(x);
    return v;
}

int first_int(const std::string& s) {  // "0" or "(0,0)"
    size_t p = s.find_first_of("-0123456789");
    if (p == std::string::npos) fail("expected an integer in '", s, "'");
    return std::stoi(s.substr(p));
}

void parse_cell_h(const fs::path& cell_h, Level& lev, int nvars) {
    auto L = read_lines(cell_h);
    const auto where = cell_h.string();
    if (L.size() < 6) fail(where, ": truncated");
    if (std::stoi(L[2]) != nvars) fail(where, ": has ", L[2], " components but Header lists ", nvars, " variables");
    lev.nghost = first_int(L[3]);
    int n = 0;
    if (std::sscanf(L[4].c_str(), " (%d", &n) != 1) fail(where, ": bad BoxArray line '", L[4], "'");
    size_t i = 5;
    for (int k = 0; k < n; ++k, ++i) {
        if (i >= L.size()) fail(where, ": truncated BoxArray");
        auto b = parse_boxes(L[i]);
        if (b.size() != 1) fail(where, ": bad box line '", L[i], "'");
        lev.boxes.push_back(b[0]);
    }
    ++i;  // ")"
    if (i >= L.size() || std::stoi(L[i]) != n) fail(where, ": FabOnDisk count mismatch");
    ++i;
    for (int k = 0; k < n; ++k, ++i) {
        char fname[1024];
        long off = 0;
        if (i >= L.size() || std::sscanf(L[i].c_str(), "FabOnDisk: %1023s %ld", fname, &off) != 2)
            fail(where, ": bad FabOnDisk line");
        lev.files.push_back((cell_h.parent_path() / fname).string());
        lev.offsets.push_back(off);
    }
}

// Read selected components of one FAB; returns per-component arrays over the FAB box (incl. ghosts).
std::vector<std::vector<double>> read_fab(const std::string& file, long off, const std::vector<int>& comps,
                                          Box& fabbox) {
    std::ifstream f(file, std::ios::binary);
    if (!f) fail("cannot open ", file);
    f.seekg(off);
    std::string h;
    std::getline(f, h);
    int nb = 0;
    if (std::sscanf(h.c_str(), "FAB ((%d,", &nb) != 1 || (nb != 4 && nb != 8))
        fail(file, "@", off, ": unsupported FAB header '", h.substr(0, 80), "'");
    const bool le = h.find("(8 7 6 5 4 3 2 1)") != std::string::npos || h.find("(4 3 2 1)") != std::string::npos;
    const bool be = h.find("(1 2 3 4 5 6 7 8)") != std::string::npos || h.find("(1 2 3 4)") != std::string::npos;
    if (!le && !be) fail(file, ": unknown byte order in FAB header");
    const bool swap = le != host_is_little_endian();
    size_t p = h.find(")))");
    if (p == std::string::npos) fail(file, ": bad FAB header");
    std::string rest = h.substr(p + 3);
    auto bx = parse_boxes(rest);
    if (bx.size() != 1) fail(file, ": bad FAB box");
    fabbox = bx[0];
    const int ncomp = std::stoi(rest.substr(rest.rfind(')') + 1));
    const size_t npt = (size_t)fabbox.n(0) * fabbox.n(1);
    const std::streamoff start = f.tellg();
    std::vector<std::vector<double>> out;
    std::vector<char> raw(npt * nb);
    for (int c : comps) {
        if (c >= ncomp) fail(file, ": component ", c, " out of range (", ncomp, ")");
        f.seekg(start + (std::streamoff)(c * npt * nb));
        f.read(raw.data(), (std::streamsize)raw.size());
        if ((size_t)f.gcount() != raw.size()) fail(file, ": short read (file truncated?)");
        if (swap)
            for (size_t k = 0; k < npt; ++k) std::reverse(raw.begin() + k * nb, raw.begin() + (k + 1) * nb);
        std::vector<double> v(npt);
        if (nb == 8) std::memcpy(v.data(), raw.data(), raw.size());
        else
            for (size_t k = 0; k < npt; ++k) {
                float x;
                std::memcpy(&x, raw.data() + k * 4, 4);
                v[k] = x;
            }
        out.push_back(std::move(v));
    }
    return out;
}

}  // namespace

bool is_plotfile(const fs::path& dir) {
    std::error_code ec;
    if (!fs::is_regular_file(dir / "Header", ec)) return false;
    std::ifstream f(dir / "Header");
    std::string first;
    std::getline(f, first);
    return first.rfind("HyperCLaw", 0) == 0;
}

std::string series_of(const fs::path& dir) {
    std::string b = dir.filename().string();
    if (b.empty()) b = dir.parent_path().filename().string();
    while (!b.empty() && std::isdigit((unsigned char)b.back())) b.pop_back();
    while (!b.empty() && (b.back() == '_' || b.back() == '-' || b.back() == '.')) b.pop_back();
    return b.empty() ? "plt" : b;
}

Plotfile read_header(const fs::path& dir) {
    Plotfile pf;
    pf.dir = dir.string();
    pf.name = dir.filename().string();
    auto L = read_lines(dir / "Header");
    const auto where = (dir / "Header").string();
    auto at = [&](size_t i) -> const std::string& {
        if (i >= L.size()) fail(where, ": truncated");
        return L[i];
    };
    if (L.empty() || L[0].rfind("HyperCLaw", 0) != 0) fail(dir.string(), " is not an AMReX plotfile");
    size_t i = 1;
    const int nv = std::stoi(at(i++));
    for (int k = 0; k < nv; ++k) pf.vars.push_back(at(i++));
    const int dim = std::stoi(at(i++));
    if (dim != 2) fail(dir.string(), ": ", dim, "D plotfiles are not supported yet (2D only)");
    pf.time = std::stod(at(i++));
    pf.finest = std::stoi(at(i++));
    auto lo = parse_nums<double>(at(i++)), hi = parse_nums<double>(at(i++));
    if (lo.size() < 2 || hi.size() < 2) fail(where, ": bad prob_lo/prob_hi");
    for (int d = 0; d < 2; ++d) { pf.prob_lo[d] = lo[d]; pf.prob_hi[d] = hi[d]; }
    pf.ref = parse_nums<int>(at(i++));  // empty line when finest == 0
    auto doms = parse_boxes(at(i++));
    auto steps = parse_nums<int>(at(i++));
    if ((int)doms.size() != pf.finest + 1 || (int)pf.ref.size() < pf.finest || (int)steps.size() < pf.finest + 1)
        fail(where, ": inconsistent level information");
    pf.levels.resize(pf.finest + 1);
    for (int l = 0; l <= pf.finest; ++l) {
        auto dx = parse_nums<double>(at(i++));
        if (dx.size() < 2) fail(where, ": bad dx line");
        pf.levels[l].dx[0] = dx[0];
        pf.levels[l].dx[1] = dx[1];
        pf.levels[l].domain = doms[l];
        pf.levels[l].step = steps[l];
    }
    pf.coord_sys = std::stoi(at(i++));
    ++i;  // boundary width
    for (int l = 0; l <= pf.finest; ++l) {
        auto hdr = parse_nums<double>(at(i++));  // "lev ngrids time"
        const int ngrids = hdr.size() >= 2 ? (int)hdr[1] : 0;
        ++i;                        // level step
        i += (size_t)ngrids * dim;  // physical extents of each grid
        std::string path = at(i++);  // e.g. "Level_0/Cell"
        if (path.find('/') == std::string::npos) path = "Level_" + std::to_string(l) + "/Cell";
        parse_cell_h(dir / (path + "_H"), pf.levels[l], nv);
    }
    if (pf.levels[0].boxes.empty()) fail(where, ": level 0 has no boxes");
    pf.itype[0] = pf.levels[0].boxes[0].type[0];
    pf.itype[1] = pf.levels[0].boxes[0].type[1];
    return pf;
}

Grid make_grid(const std::vector<Plotfile>& pfs) {
    if (pfs.empty()) fail("empty series");
    const Plotfile* deep = &pfs[0];
    for (auto& p : pfs)
        if (p.finest > deep->finest) deep = &p;
    Grid g;
    g.target_level = deep->finest;
    g.domain = deep->levels.back().domain;
    g.ref.assign(deep->ref.begin(), deep->ref.begin() + deep->finest);
    for (int d = 0; d < 2; ++d) {
        g.itype[d] = deep->itype[d];
        g.dx[d] = deep->levels.back().dx[d];
        g.prob_lo[d] = deep->prob_lo[d];
        g.prob_hi[d] = deep->prob_hi[d];
    }
    g.coord_sys = deep->coord_sys;
    for (auto& p : pfs) {
        if (p.vars != pfs[0].vars) fail(p.dir, ": variable list differs from ", pfs[0].dir);
        if (p.itype[0] != g.itype[0] || p.itype[1] != g.itype[1]) fail(p.dir, ": index type differs within series");
        for (int l = 0; l < p.finest; ++l)
            if (p.ref[l] != g.ref[l]) fail(p.dir, ": refinement ratio differs within series");
        const auto& a = p.levels[0].domain;
        const auto& b = deep->levels[0].domain;
        if (a.lo[0] != b.lo[0] || a.lo[1] != b.lo[1] || a.hi[0] != b.hi[0] || a.hi[1] != b.hi[1])
            fail(p.dir, ": level-0 domain differs within series");
    }
    return g;
}

void composite(const Plotfile& pf, const std::vector<int>& comps, const Grid& g, std::vector<double>& out,
               std::vector<unsigned char>& mask) {
    const int Nx = g.nx(), Ny = g.ny();
    const size_t npt = g.npt(), nc = comps.size();
    out.assign(npt * nc, std::numeric_limits<double>::quiet_NaN());
    mask.assign(npt, 255);
    for (int l = 0; l <= pf.finest; ++l) {
        int r = 1;
        for (int k = l; k < g.target_level; ++k) r *= g.ref[k];
        const auto& lev = pf.levels[l];
        for (size_t b = 0; b < lev.boxes.size(); ++b) {
            Box fb;
            auto fab = read_fab(lev.files[b], lev.offsets[b], comps, fb);
            const Box& vb = lev.boxes[b];
            for (int j = vb.lo[1]; j <= vb.hi[1]; ++j) {
                const int j0 = std::max(0, j * r - g.domain.lo[1]);
                const int j1 = std::min(Ny - 1, j * r + r - 1 - g.domain.lo[1]);
                for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
                    const int i0 = std::max(0, i * r - g.domain.lo[0]);
                    const int i1 = std::min(Nx - 1, i * r + r - 1 - g.domain.lo[0]);
                    const size_t src = (size_t)(j - fb.lo[1]) * fb.n(0) + (i - fb.lo[0]);
                    for (int jj = j0; jj <= j1; ++jj)
                        for (int ii = i0; ii <= i1; ++ii) {
                            const size_t dst = (size_t)jj * Nx + ii;
                            mask[dst] = (unsigned char)l;
                            for (size_t c = 0; c < nc; ++c) out[c * npt + dst] = fab[c][src];
                        }
                }
            }
        }
    }
    const size_t holes = (size_t)std::count(mask.begin(), mask.end(), (unsigned char)255);
    if (holes) fail(pf.dir, ": ", holes, " cells are not covered by any AMR level");
}

Discovery discover(const std::vector<std::string>& paths) {
    Discovery d;
    auto count = [&] {
        size_t n = 0;
        for (auto& kv : d.series) n += kv.second.size();
        return n;
    };
    auto consider = [&](const fs::path& p, bool explicit_arg) {
        const std::string name = p.filename().string();
        if (name.find(".old.") != std::string::npos) { d.skipped.emplace_back(p.string(), "AMReX backup (.old.)"); return; }
        if (is_plotfile(p)) { d.series[series_of(p)].push_back(p); return; }
        if (fs::is_regular_file(p / "Header")) d.skipped.emplace_back(p.string(), "checkpoint or non-plotfile Header");
        else if (explicit_arg) d.skipped.emplace_back(p.string(), "no Header");
    };
    for (auto& s : paths) {
        fs::path p = fs::path(s).lexically_normal();
        if (p.filename().empty()) p = p.parent_path();
        if (!fs::exists(p)) fail(s, ": no such file or directory");
        if (!fs::is_directory(p)) fail(s, ": not a directory");
        if (fs::exists(p / "Header")) { consider(p, true); continue; }
        std::vector<fs::path> kids;
        for (auto& e : fs::directory_iterator(p))
            if (e.is_directory()) kids.push_back(e.path());
        std::sort(kids.begin(), kids.end());
        const size_t before = count();
        for (auto& k : kids) consider(k, false);
        if (count() == before) d.skipped.emplace_back(p.string(), "no plotfiles inside");
    }
    for (auto& kv : d.series) {
        auto& v = kv.second;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    return d;
}

const char* coord_name(int c) { return c == 0 ? "cartesian" : c == 1 ? "RZ" : c == 2 ? "spherical" : "unknown"; }

std::string centering_name(const int t[2]) {
    if (!t[0] && !t[1]) return "cell";
    if (t[0] && t[1]) return "node";
    return t[0] ? "xface" : "yface";
}

}  // namespace pod::amrex
