"""Require a real DLAN build and execute every native FP32 physics case."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2]
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    for key in ("UIPC_MUDA_USE_DLAN", "UIPC_USE_FLOAT", "UIPC_WITH_CUDA_BACKEND"):
        if cache.get(key) != "ON":
            raise SystemExit(f"Expected {key}=ON; refusing a non-DLAN build")
    if Path(cache["CMAKE_HOME_DIRECTORY"]).resolve() != source / "tools/dlan_ci":
        raise SystemExit("Build must use this checkout's DLAN CI project")

    sdk = Path(os.environ["DLAN_SDK_ROOT"]).resolve()
    runtime_path = Path(cache["UIPC_DLAN_CURT_LIBRARY"]).resolve()
    if not runtime_path.is_relative_to(sdk):
        raise SystemExit("Build runtime does not belong to DLAN_SDK_ROOT")
    runtime = ctypes.CDLL(str(runtime_path))
    runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaGetDeviceCount.restype = ctypes.c_int
    count = ctypes.c_int()
    status = runtime.cudaGetDeviceCount(ctypes.byref(count))
    if status != 0 or count.value < 1:
        raise SystemExit(f"No usable DLAN GPU: status={status}, count={count.value}")

    backend = build / "Release/bin/libuipc_backend_cuda.so"
    executable = build / "Release/bin/dlan_physics"
    linkage = subprocess.check_output(["ldd", str(backend)], text=True)
    (output / "backend-libraries.txt").write_text(linkage)
    if str(runtime_path) not in linkage or "not found" in linkage:
        raise SystemExit(
            "Backend must load the selected DLAN runtime and all dependencies"
        )
    evidence = {
        "revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": str(source),
        "build": str(build),
        "runtime": str(runtime_path),
        "gpu_count": count.value,
        "uipc_use_float": True,
        "backend_sha256": hashlib.sha256(backend.read_bytes()).hexdigest(),
        "physics_executable_sha256": hashlib.sha256(
            executable.read_bytes()
        ).hexdigest(),
    }
    (output / "backend.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)
    report = output / "physics.xml"
    log_path = output / "physics.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            [
                "ctest",
                "--test-dir",
                str(build),
                "-R",
                "^dlan_",
                "--output-on-failure",
                "--output-junit",
                str(report),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(log_path.read_text(), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    cases = ET.parse(report).findall(".//testcase")
    expected = {
        f"dlan_{kind}_{mass}"
        for kind in ("free_fall", "force")
        for mass in ("0.01", "1", "1000")
    }
    expected.update(("dlan_initial_velocity", "dlan_stack"))
    if {case.get("name") for case in cases} != expected or len(cases) != len(expected):
        raise SystemExit("DLAN validation must execute all eight physics cases")
    if any(
        case.find(tag) is not None
        for case in cases
        for tag in ("skipped", "failure", "error")
    ):
        raise SystemExit("DLAN validation requires zero failures, errors and skips")
    print(f"libuipc DLAN validation: {len(cases)} passed, zero skipped")


if __name__ == "__main__":
    main()
