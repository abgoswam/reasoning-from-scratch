"""Keep a GPU busy enough that the cluster does not reap the job as idle.

Runs a short burst of matrix multiplications every interval, forever. Intended to
run in the background alongside `sleep infinity` on an interactive job.
"""

import argparse
import datetime
import os
import sys
import time

import torch


def _log(message: str) -> None:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
    print(f"[keepalive {stamp}] {message}", flush=True)


def _burst(device: torch.device, size: int, seconds: float) -> int:
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    deadline = time.monotonic() + seconds
    iterations = 0
    while time.monotonic() < deadline:
        c = a @ b
        a = c / (c.abs().max() + 1e-6)
        iterations += 1
    torch.cuda.synchronize(device)
    return iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--burst", type=float, default=5.0)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        _log("no CUDA device visible, refusing to start")
        sys.exit(1)

    device = torch.device(f"cuda:{args.device}")
    _log(f"pid={os.getpid()} device={torch.cuda.get_device_name(device)}")
    _log(f"interval={args.interval}s burst={args.burst}s size={args.size}")

    cycle = 0
    while True:
        cycle += 1
        started = time.monotonic()
        try:
            iterations = _burst(device, args.size, args.burst)
            mem = torch.cuda.max_memory_allocated(device) / 1024**3
            elapsed = time.monotonic() - started
            _log(f"cycle={cycle} matmuls={iterations} {elapsed:.1f}s peak_mem={mem:.1f}GiB")
        except Exception as exc:  # keep the job alive through transient CUDA errors
            _log(f"cycle={cycle} failed: {type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()

        time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
