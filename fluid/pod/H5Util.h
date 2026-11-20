// RAII handles and small helpers over the HDF5 C API.
#pragma once
#include "PodUtil.h"

#include <hdf5.h>

#include <string>
#include <utility>
#include <vector>

namespace pod::h5 {

class Handle {
  public:
    Handle() = default;
    Handle(hid_t id, herr_t (*closer)(hid_t), const std::string& what) : id_(id), close_(closer) {
        if (id < 0) fail("HDF5: cannot open/create ", what);
    }
    Handle(Handle&& o) noexcept : id_(std::exchange(o.id_, -1)), close_(o.close_) {}
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) { reset(); id_ = std::exchange(o.id_, -1); close_ = o.close_; }
        return *this;
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { reset(); }
    void reset() { if (id_ >= 0 && close_) close_(id_); id_ = -1; }
    operator hid_t() const { return id_; }

  private:
    hid_t id_ = -1;
    herr_t (*close_)(hid_t) = nullptr;
};

inline void check(herr_t e, const std::string& what) {
    if (e < 0) fail("HDF5 error: ", what);
}

template <class T> hid_t native();
template <> inline hid_t native<double>() { return H5T_NATIVE_DOUBLE; }
template <> inline hid_t native<float>() { return H5T_NATIVE_FLOAT; }
template <> inline hid_t native<int>() { return H5T_NATIVE_INT; }
template <> inline hid_t native<long long>() { return H5T_NATIVE_LLONG; }
template <> inline hid_t native<unsigned char>() { return H5T_NATIVE_UCHAR; }

inline Handle create_file(const std::string& p) {
    return Handle(H5Fcreate(p.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose, p);
}
inline Handle open_file(const std::string& p) {
    if (H5Fis_accessible(p.c_str(), H5P_DEFAULT) <= 0) fail(p, ": not an HDF5 file (or not readable)");
    return Handle(H5Fopen(p.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, p);
}
inline bool exists(hid_t loc, const std::string& path) {
    // check each component so intermediate missing groups don't raise errors
    std::string acc;
    for (auto& part : split(path, '/')) {
        acc += (acc.empty() && path[0] != '/' ? "" : "/") + part;
        if (H5Lexists(loc, acc.c_str(), H5P_DEFAULT) <= 0) return false;
    }
    return true;
}
inline Handle open_group(hid_t loc, const std::string& n) {
    if (!exists(loc, n)) fail("HDF5: no group '", n, "'");
    return Handle(H5Gopen2(loc, n.c_str(), H5P_DEFAULT), H5Gclose, n);
}
inline Handle create_group(hid_t loc, const std::string& n) {
    return Handle(H5Gcreate2(loc, n.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, n);
}
inline Handle open_dataset(hid_t loc, const std::string& n) {
    if (!exists(loc, n)) fail("HDF5: no dataset '", n, "'");
    return Handle(H5Dopen2(loc, n.c_str(), H5P_DEFAULT), H5Dclose, n);
}

inline std::vector<std::string> children(hid_t group) {
    std::vector<std::string> out;
    H5G_info_t gi;
    check(H5Gget_info(group, &gi), "group info");
    for (hsize_t i = 0; i < gi.nlinks; ++i) {
        ssize_t len = H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        std::string name(len, '\0');
        H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), len + 1, H5P_DEFAULT);
        out.push_back(name);
    }
    return out;
}

inline bool is_group(hid_t loc, const std::string& n) {
    if (!exists(loc, n)) return false;
    H5O_info2_t oi;
    if (H5Oget_info_by_name3(loc, n.c_str(), &oi, H5O_INFO_BASIC, H5P_DEFAULT) < 0) return false;
    return oi.type == H5O_TYPE_GROUP;
}

// ---- attributes ----
inline bool has_attr(hid_t obj, const char* n) { return H5Aexists(obj, n) > 0; }

inline void attr_str(hid_t obj, const char* n, const std::string& v) {
    Handle t(H5Tcopy(H5T_C_S1), H5Tclose, "string type");
    H5Tset_size(t, v.empty() ? 1 : v.size());
    H5Tset_strpad(t, H5T_STR_NULLPAD);
    Handle s(H5Screate(H5S_SCALAR), H5Sclose, "scalar space");
    Handle a(H5Acreate2(obj, n, t, s, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, n);
    check(H5Awrite(a, t, v.empty() ? "" : v.c_str()), n);
}

template <class T> void attr(hid_t obj, const char* n, const std::vector<T>& v) {
    hsize_t len = v.size();
    Handle s(H5Screate_simple(1, &len, nullptr), H5Sclose, "space");
    Handle a(H5Acreate2(obj, n, native<T>(), s, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, n);
    check(H5Awrite(a, native<T>(), v.data()), n);
}
template <class T> void attr(hid_t obj, const char* n, T v) { attr(obj, n, std::vector<T>{v}); }

inline std::string read_attr_str(hid_t obj, const char* n) {
    if (!has_attr(obj, n)) fail("HDF5: missing attribute '", n, "'");
    Handle a(H5Aopen(obj, n, H5P_DEFAULT), H5Aclose, n);
    Handle t(H5Aget_type(a), H5Tclose, n);
    if (H5Tis_variable_str(t) > 0) {
        char* p = nullptr;
        Handle mt(H5Tcopy(H5T_C_S1), H5Tclose, n);
        H5Tset_size(mt, H5T_VARIABLE);
        check(H5Aread(a, mt, &p), n);
        std::string out = p ? p : "";
        H5free_memory(p);
        return out;
    }
    std::string out(H5Tget_size(t), '\0');
    check(H5Aread(a, t, out.data()), n);
    while (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}
inline std::string read_attr_str(hid_t obj, const char* n, const std::string& def) {
    return has_attr(obj, n) ? read_attr_str(obj, n) : def;
}

template <class T> std::vector<T> read_attr(hid_t obj, const char* n) {
    if (!has_attr(obj, n)) fail("HDF5: missing attribute '", n, "'");
    Handle a(H5Aopen(obj, n, H5P_DEFAULT), H5Aclose, n);
    Handle s(H5Aget_space(a), H5Sclose, n);
    std::vector<T> v(H5Sget_simple_extent_npoints(s));
    check(H5Aread(a, native<T>(), v.data()), n);
    return v;
}

// ---- datasets ----
inline std::vector<hsize_t> dims(hid_t loc, const std::string& n) {
    Handle d = open_dataset(loc, n);
    Handle s(H5Dget_space(d), H5Sclose, n);
    std::vector<hsize_t> out(H5Sget_simple_extent_ndims(s));
    H5Sget_simple_extent_dims(s, out.data(), nullptr);
    return out;
}

// Create a dataset; chunks are one leading-index slab when rank>1 and compression>0.
inline Handle create_dataset(hid_t loc, const std::string& n, hid_t type, const std::vector<hsize_t>& dims,
                             int gzip = 0) {
    Handle s(H5Screate_simple((int)dims.size(), dims.data(), nullptr), H5Sclose, n);
    Handle dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "dcpl");
    bool nonempty = true;
    for (auto d : dims) nonempty = nonempty && d > 0;
    if (gzip > 0 && dims.size() > 1 && nonempty) {
        auto chunk = dims;
        chunk[0] = 1;
        H5Pset_chunk(dcpl, (int)chunk.size(), chunk.data());
        H5Pset_shuffle(dcpl);
        H5Pset_deflate(dcpl, (unsigned)gzip);
    }
    return Handle(H5Dcreate2(loc, n.c_str(), type, s, H5P_DEFAULT, dcpl, H5P_DEFAULT), H5Dclose, n);
}

template <class T>
void write(hid_t loc, const std::string& n, const std::vector<T>& v, const std::vector<hsize_t>& dims, int gzip = 0,
           hid_t file_type = -1) {
    Handle d = create_dataset(loc, n, file_type < 0 ? native<T>() : file_type, dims, gzip);
    if (!v.empty()) check(H5Dwrite(d, native<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()), n);
}
template <class T> void write(hid_t loc, const std::string& n, const std::vector<T>& v) {
    write(loc, n, v, {(hsize_t)v.size()});
}

// Write `buf` (dense, shape = count) into the hyperslab start/count of dataset d.
template <class T>
void write_slab(hid_t d, const std::vector<hsize_t>& start, const std::vector<hsize_t>& count, const T* buf) {
    Handle fs(H5Dget_space(d), H5Sclose, "space");
    check(H5Sselect_hyperslab(fs, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr), "select");
    Handle ms(H5Screate_simple((int)count.size(), count.data(), nullptr), H5Sclose, "mem space");
    check(H5Dwrite(d, native<T>(), ms, fs, H5P_DEFAULT, buf), "write slab");
}

template <class T> std::vector<T> read(hid_t loc, const std::string& n) {
    Handle d = open_dataset(loc, n);
    Handle s(H5Dget_space(d), H5Sclose, n);
    std::vector<T> v(H5Sget_simple_extent_npoints(s));
    if (!v.empty()) check(H5Dread(d, native<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()), n);
    return v;
}

// Read a strided hyperslab (start/stride/count per dimension) into a dense buffer.
template <class T>
std::vector<T> read_slab(hid_t loc, const std::string& n, const std::vector<hsize_t>& start,
                         const std::vector<hsize_t>& stride, const std::vector<hsize_t>& count) {
    Handle d = open_dataset(loc, n);
    Handle fs(H5Dget_space(d), H5Sclose, n);
    check(H5Sselect_hyperslab(fs, H5S_SELECT_SET, start.data(), stride.data(), count.data(), nullptr), n);
    Handle ms(H5Screate_simple((int)count.size(), count.data(), nullptr), H5Sclose, n);
    size_t total = 1;
    for (auto c : count) total *= c;
    std::vector<T> v(total);
    if (total) check(H5Dread(d, native<T>(), ms, fs, H5P_DEFAULT, v.data()), n);
    return v;
}

}  // namespace pod::h5
