// Initialize and export one grid. Numerical time evolution is still pending.
#include "RunOutput.h"
#include <exception>
#include <iostream>
#include <string_view>

using namespace std;

int main(int argc, char* argv[]) {
    const auto usage = [&] {
        cout << "Usage: " << argv[0] << " [width height [pattern]]\n"
             << "       " << argv[0] << " --config path.cfg\n"
             << "Patterns: still, gradient (default), center, inward, vortex\n"
             << "Creates a fresh dated folder under ./saves/ by default.\n"
             << "Initialization only: dt/duration are recorded, no CFD solver yet.\n";
    };
    try {
        if (argc == 2 && string_view(argv[1]) == "--help") {
            usage();
            return 0;
        }
        RunConfig config;
        if (argc == 3 && string_view(argv[1]) == "--config") {
            config = read_config(argv[2]);
        } else if (argc == 3 || argc == 4) {
            config.width = parse_size(argv[1]);
            config.height = parse_size(argv[2]);
            if (argc == 4) config.pattern = parse_pattern(argv[3]);
        } else if (argc != 1) {
            usage();
            return 1;
        }
        config.validate();
        cout << "Grid: " << config.width << " x " << config.height
             << ", pattern: " << pattern_name(config.pattern)
             << ", domain: " << config.domain.length_x << " x "
             << config.domain.length_y << " m, periodic, cell-centered\n";
        const auto directory = export_initial_state(config);
        cout << "Saved initial state and recipe to " << directory.string() << '\n'
             << "Time: 0 s; solver not implemented.\n";
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
