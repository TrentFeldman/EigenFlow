"""Report installed versions/allocation; optionally require a real NVIDIA check.

The CUDA option fails on a ROCm build even though both backends expose the
same torch.cuda API. It exercises forward/backward GPU operations and compares
small CPU/GPU matrix products; full model smoke testing remains a separate gate.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from .benchmark import hardware


def main(argv=None) -> int:
    """Print environment JSON, optionally save it, and return 1 if unready.

    --require-cuda requires an NVIDIA build, visible device and successful
    forward/backward numerical probe. It never installs or changes packages.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    report = {"executable": sys.executable, "python": sys.version.split()[0],
              "in_virtual_environment": sys.prefix != sys.base_prefix,
              "hardware": hardware(), "dependencies": {}}
    ready = report['in_virtual_environment'] and sys.version_info >= (3, 11)
    for name in ('numpy', 'scipy', 'torch'):
        try:
            module = __import__(name)
            report['dependencies'][name] = module.__version__
            if name == 'torch':
                report.update(gpu_available=module.cuda.is_available(),
                              cuda_build=module.version.cuda, hip_build=module.version.hip)
        except ImportError as error:
            report['dependencies'][name] = str(error)
            ready = False
    if args.require_cuda:
        try:
            import torch
            if not torch.version.cuda or torch.version.hip or not torch.cuda.is_available():
                raise RuntimeError('NVIDIA CUDA build and visible allocated GPU required')
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.use_deterministic_algorithms(True)
            cpu = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
            gpu = cpu.to('cuda').requires_grad_()
            output = gpu @ gpu.T
            torch.testing.assert_close(output.cpu(), cpu @ cpu.T, rtol=1e-5, atol=1e-6)
            output.square().mean().backward()
            if not torch.isfinite(gpu.grad).all():
                raise RuntimeError('Nonfinite CUDA gradient')
            torch.cuda.synchronize()
            report['cuda_probe'] = {"passed": True, "name": torch.cuda.get_device_name(),
                                    "capability": torch.cuda.get_device_capability(),
                                    "compiled_architectures": torch.cuda.get_arch_list(),
                                    "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        except (ImportError, RuntimeError, AssertionError) as error:
            report['cuda_probe'] = {"passed": False, "error": str(error)}
            ready = False
    report['ready'] = ready
    serialized = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            output.write(serialized)
    print(serialized, end='')
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
