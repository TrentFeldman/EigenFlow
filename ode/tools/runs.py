"""List trained runs and generated datasets in one table.

    python -m ode runs          current runs in ode/runs/ plus every dataset
    python -m ode runs --all    also the finished history in ode/runs/archive/

Move a run you no longer use into ode/runs/archive/ to drop it from the
default list; nothing reads runs from fixed paths, so moving is safe.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _size(path):
    """Human-readable total size of every file under path."""
    total = float(sum(file.stat().st_size for file in path.rglob("*") if file.is_file()))
    for unit in ("B", "K", "M", "G"):
        if total < 1024 or unit == "G":
            return f"{total:.0f}{unit}" if unit == "B" else f"{total:.1f}{unit}"
        total /= 1024


def _runs(folder):
    """Run folders directly inside folder, oldest name first."""
    return sorted(path for path in folder.iterdir() if path.is_dir() and path.name != "archive") if folder.exists() else []


def main(argv=None):
    """Print the run and dataset tables; return 0."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Include ode/runs/archive/")
    args = parser.parse_args(argv)
    current, archived = _runs(ROOT / "runs"), _runs(ROOT / "runs" / "archive")
    used_by = {}
    print(f"{'run':<44} {'status':<10} {'dataset':<20} {'epochs':>6} {'avg error':>9} {'size':>6}")
    for run in current + archived:
        record = run / "run.json"
        report = json.loads(record.read_text()) if record.exists() else {}
        dataset = Path(report["data"]).name if "data" in report else "-"
        used_by.setdefault(dataset, []).append(run.name)
        if run in archived and not args.all:
            continue
        best = report.get("best_validation_relative_l2")
        print(f"{str(run.relative_to(ROOT / 'runs')):<44} {report.get('status', 'record'):<10} {dataset:<20} "
              f"{len(report.get('epochs', [])) or '-':>6} {f'{best:.2%}' if best is not None else '-':>9} {_size(run):>6}")
    if archived and not args.all:
        print(f"({len(archived)} more in ode/runs/archive/; add --all to list them)")
    print(f"\n{'dataset':<26} {'equations':>9} {'status':<10} {'size':>6}  used by")
    for data in sorted(path for path in (ROOT / "data").iterdir() if path.is_dir()):
        manifest_path = data / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        users = used_by.get(data.name, [])
        print(f"{data.name:<26} {f"{manifest['succeeded']:,}" if 'succeeded' in manifest else '-':>9} {manifest.get('status', '?'):<10} {_size(data):>6}  "
              f"{len(users)} run(s)" + (f": {', '.join(users)}" if 0 < len(users) <= 2 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
