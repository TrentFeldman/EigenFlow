// Top-level command dispatch for fluid_pod.
#include "Commands.h"
#include "PodUtil.h"

#include <hdf5.h>

#include <cstdio>
#include <cstring>

namespace pod {

namespace {
const char* kUsage =
    "usage: fluid_pod <command> [options] ...\n"
    "\n"
    "Turn 2D AMReX plotfile output into HDF5 snapshots and reduce them with POD.\n"
    "\n"
    "commands:\n"
    "  info         describe a run directory, a plotfile, or a fluid_pod HDF5 file\n"
    "  convert      plotfile series -> one HDF5 file on a uniform grid (+ ParaView .xmf)\n"
    "  pod          proper orthogonal decomposition of one series (LAPACK SVD)\n"
    "  reconstruct  rebuild snapshots from K modes; --compare reports the error\n"
    "  export       dump HDF5 datasets to NumPy .npy files\n"
    "\n"
    "typical workflow:\n"
    "  fluid_pod info    ~/dat/run_000\n"
    "  fluid_pod convert ~/dat/run_000 -s nddataPLT -o run_000.h5\n"
    "  fluid_pod pod     run_000.h5 -v x_velocity,y_velocity -e 0.999\n"
    "  fluid_pod reconstruct run_000.nddataPLT.pod.h5 -k 5 --compare\n"
    "\n"
    "'fluid_pod <command> --help' shows every option. See fluid/pod/README.md.\n";
}

int run_cli(const std::vector<std::string>& args) {
    try {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);  // errors are reported as exceptions instead
        if (args.empty() || args[0] == "-h" || args[0] == "--help" || args[0] == "help") {
            if (args.size() >= 2) return run_cli({args[1], "--help"});
            std::fputs(kUsage, stdout);
            return args.empty() ? 1 : 0;
        }
        if (args[0] == "--version" || args[0] == "version") {
            unsigned maj, min, rel;
            H5get_libversion(&maj, &min, &rel);
            std::printf("fluid_pod %s (HDF5 %u.%u.%u)\n", version(), maj, min, rel);
            return 0;
        }
        const std::vector<std::string> rest(args.begin() + 1, args.end());
        const std::string& cmd = args[0];
        if (cmd == "info") return cmd_info(rest);
        if (cmd == "convert") return cmd_convert(rest);
        if (cmd == "pod") return cmd_pod(rest);
        if (cmd == "reconstruct" || cmd == "rec") return cmd_reconstruct(rest);
        if (cmd == "export") return cmd_export(rest);
        fail("unknown command '", cmd, "' (commands: info, convert, pod, reconstruct, export)");
    } catch (const HelpShown&) {
        return 0;
    } catch (const Error& e) {
        std::fprintf(stderr, "fluid_pod: error: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fluid_pod: internal error: %s\n", e.what());
        return 2;
    }
}

}  // namespace pod
