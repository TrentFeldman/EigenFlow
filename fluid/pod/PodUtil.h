// Shared helpers for the POD toolkit: errors, logging, strings, globs, slices, CLI parsing.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pod {

inline const char* version() { return "0.1.0"; }

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct HelpShown {};  // thrown after --help output; exit code 0

template <class... A> std::string cat(const A&... a) {
    std::ostringstream o;
    (o << ... << a);
    return o.str();
}
template <class... A> [[noreturn]] void fail(const A&... a) { throw Error(cat(a...)); }

inline int& verbosity() { static int v = 1; return v; }  // 0 quiet, 1 normal, 2 verbose
inline void info(const std::string& s) { if (verbosity() >= 1) std::fprintf(stderr, "%s\n", s.c_str()); }
inline void debug(const std::string& s) { if (verbosity() >= 2) std::fprintf(stderr, "%s\n", s.c_str()); }
inline void warn(const std::string& s) { std::fprintf(stderr, "warning: %s\n", s.c_str()); }

inline bool host_is_little_endian() {
    const std::uint16_t one = 1;
    return *reinterpret_cast<const unsigned char*>(&one) == 1;
}

inline std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(d, a);
        if (b == std::string::npos) b = s.size();
        if (b > a) v.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return v;
}

inline std::string join(const std::vector<std::string>& v, const std::string& sep = ",") {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) out += (i ? sep : "") + v[i];
    return out;
}

// Shell-style glob with '*' and '?' (portable; no fnmatch).
inline bool glob_match(const std::string& pattern, const std::string& text) {
    const char *p = pattern.c_str(), *s = text.c_str(), *star = nullptr, *mark = nullptr;
    while (*s) {
        if (*p == '?' || *p == *s) { ++p; ++s; }
        else if (*p == '*') { star = p++; mark = s; }
        else if (star) { p = star + 1; s = ++mark; }
        else return false;
    }
    while (*p == '*') ++p;
    return !*p;
}

// Comma-separated globs, selected in pattern order then source order. Empty selects all.
inline std::vector<std::string> select_names(const std::vector<std::string>& all, const std::string& patterns,
                                             const std::string& what, bool allow_unmatched = false) {
    if (patterns.empty()) return all;
    std::vector<std::string> out;
    for (auto& p : split(patterns, ',')) {
        bool hit = false;
        for (auto& n : all)
            if (glob_match(p, n)) {
                hit = true;
                if (std::find(out.begin(), out.end(), n) == out.end()) out.push_back(n);
            }
        if (!hit && !allow_unmatched) fail("no ", what, " matches '", p, "'. Available: ", join(all, ", "));
    }
    return out;
}

inline std::optional<long> parse_long(const std::string& t, const std::string& ctx) {
    if (t.empty()) return std::nullopt;
    try {
        size_t p = 0;
        long v = std::stol(t, &p);
        if (p == t.size()) return v;
    } catch (...) {
    }
    fail("bad integer '", t, "' in '", ctx, "'");
}

inline std::optional<double> parse_double(const std::string& t, const std::string& ctx) {
    if (t.empty()) return std::nullopt;
    try {
        size_t p = 0;
        double v = std::stod(t, &p);
        if (p == t.size()) return v;
    } catch (...) {
    }
    fail("bad number '", t, "' in '", ctx, "'");
}

// Regular index selection start + i*step, i < count.
struct Slice {
    int start = 0, step = 1, count = 0;
    int at(int i) const { return start + i * step; }
};

// Python-style "start:stop:step" over n items (stop exclusive, negatives from the end, step > 0).
// A single integer selects one item; empty selects all.
inline Slice parse_slice(const std::string& s, int n) {
    if (s.empty()) return {0, 1, n};
    std::vector<std::string> parts;
    for (size_t a = 0;;) {
        size_t b = s.find(':', a);
        parts.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    auto wrap = [&](long v) { return v < 0 ? v + n : v; };
    if (parts.size() == 1) {
        auto v = parse_long(parts[0], s);
        if (!v) fail("empty slice");
        long i = wrap(*v);
        if (i < 0 || i >= n) fail("index ", s, " out of range (have ", n, ")");
        return {int(i), 1, 1};
    }
    if (parts.size() > 3) fail("bad slice '", s, "' (expected start:stop[:step])");
    long step = parts.size() == 3 ? parse_long(parts[2], s).value_or(1) : 1;
    if (step <= 0) fail("slice step must be positive in '", s, "'");
    long a = std::clamp(wrap(parse_long(parts[0], s).value_or(0)), 0L, (long)n);
    long b = std::clamp(wrap(parse_long(parts[1], s).value_or(n)), 0L, (long)n);
    int count = b > a ? int((b - a + step - 1) / step) : 0;
    if (count == 0) fail("slice '", s, "' selects nothing (have ", n, ")");
    return {int(a), int(step), count};
}

// Inclusive numeric range "A:B", either side optional; "A" alone means exactly A.
struct Range {
    std::optional<double> lo, hi;
    bool contains(double v) const { return (!lo || v >= *lo) && (!hi || v <= *hi); }
};
inline Range parse_range(const std::string& s) {
    Range r;
    if (s.empty()) return r;
    auto c = s.find(':');
    if (c == std::string::npos) { r.lo = r.hi = parse_double(s, s); return r; }
    r.lo = parse_double(s.substr(0, c), s);
    r.hi = parse_double(s.substr(c + 1), s);
    return r;
}

inline std::string human_bytes(double b) {
    const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int i = 0;
    while (b >= 1024 && i < 4) { b /= 1024; ++i; }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f %s", b, u[i]);
    return buf;
}

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

inline int default_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 1;
}

// Run f(i) for i in [0, n) on up to `threads` threads; rethrows the first exception.
template <class F> void parallel_for(size_t n, int threads, F f) {
    std::atomic<size_t> next{0};
    std::exception_ptr err;
    std::mutex m;
    auto work = [&] {
        for (;;) {
            size_t i = next++;
            if (i >= n) return;
            try {
                f(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(m);
                if (!err) err = std::current_exception();
            }
        }
    };
    const size_t nt = std::min(n, (size_t)std::max(1, threads));
    if (nt <= 1) {
        work();
    } else {
        std::vector<std::thread> pool;
        for (size_t k = 0; k < nt; ++k) pool.emplace_back(work);
        for (auto& t : pool) t.join();
    }
    if (err) std::rethrow_exception(err);
}

// ---------------------------------------------------------------------------------------------
// Minimal CLI parser: --long VALUE, --long=VALUE, -s VALUE, flags, positionals, "--" terminator.
struct Opt {
    std::string name;    // long name without dashes
    char short_name;     // 0 if none
    std::string meta;    // value placeholder; empty => flag
    std::string help;
};

struct Args {
    std::vector<std::string> pos;
    std::map<std::string, std::string> values;
    std::set<std::string> flags;
    bool has(const std::string& n) const { return flags.count(n) || values.count(n); }
    std::string get(const std::string& n, const std::string& def = "") const {
        auto it = values.find(n);
        return it == values.end() ? def : it->second;
    }
    long get_long(const std::string& n, long def) const { return has(n) ? *parse_long(get(n), "--" + n) : def; }
    double get_double(const std::string& n, double def) const { return has(n) ? *parse_double(get(n), "--" + n) : def; }
};

inline std::string format_help(const std::string& usage, const std::string& about, const std::vector<Opt>& opts) {
    std::string s = "usage: " + usage + "\n\n" + about + "\n\noptions:\n";
    for (auto& o : opts) {
        std::string lhs = "  ";
        lhs += o.short_name ? std::string("-") + o.short_name + ", " : "    ";
        lhs += "--" + o.name + (o.meta.empty() ? "" : " " + o.meta);
        if (lhs.size() < 30) lhs.resize(30, ' ');
        else lhs += "\n" + std::string(30, ' ');
        s += lhs + o.help + "\n";
    }
    s += "  -h, --help                  show this help\n";
    return s;
}

inline Args parse_args(const std::vector<std::string>& argv, const std::string& usage, const std::string& about,
                       const std::vector<Opt>& opts) {
    Args a;
    auto find_long = [&](const std::string& n) -> const Opt* {
        for (auto& o : opts) if (o.name == n) return &o;
        return nullptr;
    };
    auto find_short = [&](char c) -> const Opt* {
        for (auto& o : opts) if (o.short_name == c) return &o;
        return nullptr;
    };
    bool only_pos = false;
    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& t = argv[i];
        if (only_pos || t.size() < 2 || t[0] != '-' || (t[1] >= '0' && t[1] <= '9') || t[1] == '.') {
            a.pos.push_back(t);
            continue;
        }
        if (t == "--") { only_pos = true; continue; }
        if (t == "-h" || t == "--help") {
            std::fputs(format_help(usage, about, opts).c_str(), stdout);
            throw HelpShown{};
        }
        const Opt* o = nullptr;
        std::optional<std::string> inline_val;
        if (t[1] == '-') {
            std::string n = t.substr(2);
            auto eq = n.find('=');
            if (eq != std::string::npos) { inline_val = n.substr(eq + 1); n = n.substr(0, eq); }
            o = find_long(n);
            if (!o) fail("unknown option '--", n, "' (see --help)");
        } else {
            if (t.size() != 2) fail("unknown option '", t, "' (short options take a separate value; see --help)");
            o = find_short(t[1]);
            if (!o) fail("unknown option '", t, "' (see --help)");
        }
        if (o->meta.empty()) {
            if (inline_val) fail("option --", o->name, " takes no value");
            a.flags.insert(o->name);
        } else {
            if (inline_val) a.values[o->name] = *inline_val;
            else if (i + 1 < argv.size()) a.values[o->name] = argv[++i];
            else fail("option --", o->name, " needs a value ", o->meta);
        }
    }
    return a;
}

}  // namespace pod
