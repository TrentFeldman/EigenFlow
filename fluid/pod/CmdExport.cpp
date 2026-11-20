// fluid_pod export: dump every dataset of an HDF5 file (or one group) to NumPy .npy files,
// plus attributes.json, so data can be used from numpy without h5py.
#include "Commands.h"
#include "H5Util.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace pod {

namespace fs = std::filesystem;

namespace {

struct DsetInfo {
    std::string path;
    std::vector<hsize_t> dims;
    std::string descr;  // numpy dtype, e.g. <f8
    hid_t mem_type;
    size_t elem;
};

bool describe(hid_t file, const std::string& path, DsetInfo& d) {
    h5::Handle ds = h5::open_dataset(file, path);
    h5::Handle t(H5Dget_type(ds), H5Tclose, path);
    const H5T_class_t cls = H5Tget_class(t);
    const size_t sz = H5Tget_size(t);
    d.path = path;
    d.dims = h5::dims(file, path);
    if (cls == H5T_FLOAT && (sz == 4 || sz == 8)) {
        d.descr = sz == 8 ? "<f8" : "<f4";
        d.mem_type = sz == 8 ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    } else if (cls == H5T_INTEGER && (sz == 1 || sz == 2 || sz == 4 || sz == 8)) {
        const bool sgn = H5Tget_sign(t) == H5T_SGN_2;
        d.descr = std::string(sz == 1 ? "|" : "<") + (sgn ? "i" : "u") + std::to_string(sz);
        switch (sz) {
            case 1: d.mem_type = sgn ? H5T_NATIVE_SCHAR : H5T_NATIVE_UCHAR; break;
            case 2: d.mem_type = sgn ? H5T_NATIVE_SHORT : H5T_NATIVE_USHORT; break;
            case 4: d.mem_type = sgn ? H5T_NATIVE_INT32 : H5T_NATIVE_UINT32; break;
            default: d.mem_type = sgn ? H5T_NATIVE_INT64 : H5T_NATIVE_UINT64; break;
        }
    } else {
        return false;
    }
    d.elem = sz;
    return true;
}

void write_npy(const fs::path& p, const DsetInfo& d, const std::vector<char>& data) {
    std::string shape = "(";
    for (size_t i = 0; i < d.dims.size(); ++i) shape += std::to_string(d.dims[i]) + (d.dims.size() == 1 ? "," : i + 1 < d.dims.size() ? ", " : "");
    shape += ")";
    std::string hdr = "{'descr': '" + d.descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    const size_t total = 10 + hdr.size() + 1;
    hdr += std::string((64 - total % 64) % 64, ' ') + "\n";
    std::ofstream o(p, std::ios::binary);
    if (!o) fail("cannot write ", p.string());
    const unsigned short len = (unsigned short)hdr.size();
    o.write("\x93NUMPY\x01\x00", 8);
    const char lb[2] = {char(len & 0xff), char(len >> 8)};
    o.write(lb, 2);
    o << hdr;
    o.write(data.data(), (std::streamsize)data.size());
    if (!o) fail("error writing ", p.string());
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

std::string attrs_json(hid_t obj) {
    std::string out = "{";
    H5O_info2_t oi;
    h5::check(H5Oget_info3(obj, &oi, H5O_INFO_NUM_ATTRS), "object info");
    const hsize_t na = oi.num_attrs;
    for (hsize_t i = 0; i < na; ++i) {
        h5::Handle at(H5Aopen_by_idx(obj, ".", H5_INDEX_NAME, H5_ITER_INC, i, H5P_DEFAULT, H5P_DEFAULT), H5Aclose,
                      "attribute");
        ssize_t nl = H5Aget_name(at, 0, nullptr);
        std::string name(nl, '\0');
        H5Aget_name(at, nl + 1, name.data());
        h5::Handle t(H5Aget_type(at), H5Tclose, name);
        std::string val;
        if (H5Tget_class(t) == H5T_STRING) {
            val = "\"" + json_escape(h5::read_attr_str(obj, name.c_str())) + "\"";
        } else {
            h5::Handle sp(H5Aget_space(at), H5Sclose, name);
            std::vector<double> v(H5Sget_simple_extent_npoints(sp));
            if (H5Aread(at, H5T_NATIVE_DOUBLE, v.data()) < 0) continue;
            val = v.size() == 1 ? "" : "[";
            for (size_t k = 0; k < v.size(); ++k) {
                char b[40];
                std::snprintf(b, sizeof b, "%.17g", v[k]);
                val += (k ? ", " : "") + std::string(b);
            }
            if (v.size() != 1) val += "]";
        }
        out += std::string(i ? ", " : "") + "\"" + json_escape(name) + "\": " + val;
    }
    return out + "}";
}

void walk(hid_t file, const std::string& path, std::vector<std::string>& groups, std::vector<std::string>& dsets) {
    groups.push_back(path);
    h5::Handle g = h5::open_group(file, path);
    for (auto& c : h5::children(g)) {
        const std::string p = (path == "/" ? "" : path) + "/" + c;
        if (h5::is_group(file, p)) walk(file, p, groups, dsets);
        else dsets.push_back(p);
    }
}

}  // namespace

int cmd_export(const std::vector<std::string>& argv) {
    const std::vector<Opt> opts = {
        {"dir", 'd', "DIR", "output directory (default: <file-stem>_npy)"},
        {"path", 'p', "GROUP", "only export below this group, e.g. /nddataPLT"},
        {"only", 0, "GLOBS", "only datasets whose full path matches, e.g. '*/x_velocity,*/time'"},
        {"list", 'l', "", "list datasets (path, shape, dtype) and exit"},
        {"force", 'f', "", "write into an existing directory"},
        {"quiet", 'q', "", "only print errors"},
    };
    auto a = parse_args(argv, "fluid_pod export [options] FILE.h5",
                        "Export datasets to NumPy .npy files (np.load works without h5py). Each dataset\n"
                        "/a/b/c becomes a.b.c.npy; attributes of every group go to attributes.json.",
                        opts);
    if (a.has("quiet")) verbosity() = 0;
    if (a.pos.size() != 1) fail("expected exactly one HDF5 file (see --help)");
    const std::string in = a.pos[0];
    h5::Handle f = h5::open_file(in);
    std::string root = a.get("path", "/");
    if (root.empty() || root[0] != '/') root = "/" + root;
    if (root.size() > 1 && root.back() == '/') root.pop_back();
    std::vector<std::string> groups, dsets;
    walk(f, root, groups, dsets);
    if (a.has("only")) {
        std::vector<std::string> keep;
        for (auto& d : dsets)
            for (auto& pat : split(a.get("only"), ','))
                if (glob_match(pat, d)) { keep.push_back(d); break; }
        dsets = keep;
    }

    if (a.has("list")) {
        for (auto& p : dsets) {
            DsetInfo d;
            const bool ok = describe(f, p, d);
            std::string shape;
            for (size_t i = 0; i < d.dims.size(); ++i) shape += (i ? "x" : "") + std::to_string(d.dims[i]);
            std::printf("%-50s %-16s %s\n", p.c_str(), shape.c_str(), ok ? d.descr.c_str() : "(unsupported type)");
        }
        return 0;
    }

    fs::path dir = a.has("dir") ? fs::path(a.get("dir")) : fs::path(fs::path(in).stem().string() + "_npy");
    if (fs::exists(dir) && !a.has("force")) fail(dir.string(), " already exists (use -f to write into it)");
    fs::create_directories(dir);
    double bytes = 0;
    int count = 0;
    for (auto& p : dsets) {
        DsetInfo d;
        if (!describe(f, p, d)) { warn(p + ": unsupported datatype, skipped"); continue; }
        size_t n = 1;
        for (auto x : d.dims) n *= x;
        std::vector<char> buf(n * d.elem);
        h5::Handle ds = h5::open_dataset(f, p);
        if (n) h5::check(H5Dread(ds, d.mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()), p);
        std::string name = p.substr(1);
        std::replace(name.begin(), name.end(), '/', '.');
        write_npy(dir / (name + ".npy"), d, buf);
        bytes += (double)buf.size();
        ++count;
    }
    std::ofstream j(dir / "attributes.json");
    j << "{\n";
    for (size_t i = 0; i < groups.size(); ++i) {
        h5::Handle g = h5::open_group(f, groups[i]);
        j << "  \"" << json_escape(groups[i]) << "\": " << attrs_json(g) << (i + 1 < groups.size() ? ",\n" : "\n");
    }
    j << "}\n";
    if (verbosity() >= 1) std::printf("exported %d datasets (%s) to %s/\n", count, human_bytes(bytes).c_str(), dir.string().c_str());
    return 0;
}

}  // namespace pod
