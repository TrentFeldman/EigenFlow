"""Report interpreter and dependency readiness for the ODE pipeline.

Run this setup check before generation, training or prediction to inspect the
active Python environment, required library versions and GPU availability.
"""

import json
import sys


def main() -> int:
    """Print environment readiness as a JSON report.

    Returns:
        Exit code 0 for Python >= 3.11 in a virtual environment with all required
        imports available, otherwise 1. GPU availability is reported separately.
    """
    report = {
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "in_virtual_environment": sys.prefix != sys.base_prefix,
        "dependencies": {},
    }
    ready = report["in_virtual_environment"] and sys.version_info >= (3, 11)

    for name in ("numpy", "scipy", "torch"):
        try:
            module = __import__(name)
            report["dependencies"][name] = module.__version__

            if name == "torch":
                report["gpu_available"] = module.cuda.is_available()
        except ImportError as error:
            report["dependencies"][name] = str(error)
            ready = False

    report["ready"] = ready
    print(json.dumps(report, indent=2))

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
