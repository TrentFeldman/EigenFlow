#include "VtkHdf.h"

#include "H5Util.h"

namespace pod {

namespace fs = std::filesystem;

void write_vtkhdf(const fs::path& sidecar, const fs::path& target, const std::vector<double>& steps,
                  const VtkGrid& g, const std::vector<VtkField>& fields) {
    std::error_code ec;
    const bool same_dir = fs::absolute(target, ec).parent_path() == fs::absolute(sidecar, ec).parent_path();
    // HDF5 resolves relative external links against the directory of the linking file.
    const std::string link_file = same_dir ? target.filename().string() : fs::absolute(target).string();
    const long long nt = (long long)steps.size();

    h5::Handle f = h5::create_file(sidecar.string());
    h5::Handle root = h5::create_group(f, "VTKHDF");
    h5::attr(root, "Version", std::vector<long long>{2, 2});
    {   // fixed-length, null-terminated ASCII, as VTK writes it
        h5::Handle t(H5Tcopy(H5T_C_S1), H5Tclose, "type");
        H5Tset_size(t, 9);
        H5Tset_strpad(t, H5T_STR_NULLTERM);
        h5::Handle s(H5Screate(H5S_SCALAR), H5Sclose, "space");
        h5::Handle a(H5Acreate2(root, "Type", t, s, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, "Type");
        h5::check(H5Awrite(a, t, "ImageData"), "Type");
    }
    const int ex = g.node_centered ? g.nx - 1 : g.nx, ey = g.node_centered ? g.ny - 1 : g.ny;
    h5::attr(root, "WholeExtent", std::vector<int>{0, ex, 0, ey, 0, 0});
    h5::attr(root, "Origin", std::vector<double>{g.origin[0], g.origin[1], 0.0});
    h5::attr(root, "Spacing", std::vector<double>{g.spacing[0], g.spacing[1], 1.0});
    h5::attr(root, "Direction", std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});

    const char* data_group = g.node_centered ? "PointData" : "CellData";
    const char* offs_group = g.node_centered ? "PointDataOffsets" : "CellDataOffsets";
    h5::Handle data = h5::create_group(root, data_group);
    h5::Handle other = h5::create_group(root, g.node_centered ? "CellData" : "PointData");
    h5::Handle st = h5::create_group(root, "Steps");
    h5::attr(st, "NSteps", nt);
    h5::write(st, "Values", steps);
    h5::Handle offs = h5::create_group(st, offs_group);
    h5::Handle offs_other = h5::create_group(st, g.node_centered ? "CellDataOffsets" : "PointDataOffsets");

    std::vector<long long> per_step(nt), zeros(nt, 0);
    for (long long t = 0; t < nt; ++t) per_step[t] = t;
    for (auto& fld : fields) {
        if (fld.inline_.empty()) {
            h5::check(H5Lcreate_external(link_file.c_str(), fld.dataset.c_str(), data, fld.name.c_str(), H5P_DEFAULT,
                                         H5P_DEFAULT),
                      "external link " + fld.name);
            h5::write(offs, fld.name, per_step);
        } else {
            h5::write(data, fld.name, fld.inline_, {1, (hsize_t)g.ny, (hsize_t)g.nx});
            h5::write(offs, fld.name, zeros);
        }
    }
}

}  // namespace pod
