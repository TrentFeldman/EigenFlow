// Reader for 2D AMReX plotfiles ("HyperCLaw-V1.1" Header + native FAB data).
//
// Supported: any number of AMR levels/grids, any refinement ratio, ghost cells, float32/float64,
// either byte order, cell/node/face index types, non-zero domain origin, and series whose finest
// level changes over time. Every snapshot is composited onto one uniform grid at the series'
// finest resolution (piecewise-constant injection from coarse levels; finer levels overwrite).
// Not supported: 1D/3D, checkpoint directories (chk*), particle data.
#pragma once
#include "PodUtil.h"

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace pod::amrex {

namespace fs = std::filesystem;

struct Box {
    int lo[2]{}, hi[2]{}, type[2]{};
    int n(int d) const { return hi[d] - lo[d] + 1; }
};

struct Level {
    Box domain;  // cell-centred domain box
    double dx[2]{};
    int step = 0;
    int nghost = 0;
    std::vector<Box> boxes;          // valid boxes, in the data's index space
    std::vector<std::string> files;  // Cell_D_* file holding each box
    std::vector<long> offsets;       // byte offset of each FAB inside its file
};

struct Plotfile {
    std::string dir, name;
    std::vector<std::string> vars;
    double time = 0;
    int finest = 0;
    int coord_sys = 0;  // 0 cartesian, 1 RZ, 2 spherical
    int itype[2]{};     // 0 cell-centred, 1 nodal (per direction)
    double prob_lo[2]{}, prob_hi[2]{};
    std::vector<int> ref;  // ref[l] = ratio between level l and l+1
    std::vector<Level> levels;
    int step() const { return levels.empty() ? 0 : levels[0].step; }
};

// Uniform target grid shared by all snapshots of a series.
struct Grid {
    Box domain;  // cell-centred domain at the target level
    int target_level = 0;
    std::vector<int> ref;
    int itype[2]{};
    double dx[2]{}, prob_lo[2]{}, prob_hi[2]{};
    int coord_sys = 0;
    int nx() const { return domain.n(0) + itype[0]; }
    int ny() const { return domain.n(1) + itype[1]; }
    size_t npt() const { return (size_t)nx() * ny(); }
    // physical coordinate of array index i along d (index 0 = domain.lo, whose lower face is prob_lo)
    double coord(int d, int i) const { return prob_lo[d] + (i + (itype[d] ? 0.0 : 0.5)) * dx[d]; }
};

bool is_plotfile(const fs::path& dir);
std::string series_of(const fs::path& dir);  // plt00120 -> plt, nddataPLT00000030 -> nddataPLT
Plotfile read_header(const fs::path& dir);
Grid make_grid(const std::vector<Plotfile>& series);

// Composite the given components onto grid g.
// out: [ncomp][ny][nx] (x fastest); mask: [ny][nx] = AMR level that supplied each value.
void composite(const Plotfile& pf, const std::vector<int>& comps, const Grid& g, std::vector<double>& out,
               std::vector<unsigned char>& mask);

struct Discovery {
    std::map<std::string, std::vector<fs::path>> series;     // name -> plotfile dirs (sorted)
    std::vector<std::pair<std::string, std::string>> skipped;  // path, reason
};
// Paths may be plotfile directories or run directories containing them (one level deep).
Discovery discover(const std::vector<std::string>& paths);

const char* coord_name(int coord_sys);
std::string centering_name(const int itype[2]);

}  // namespace pod::amrex
