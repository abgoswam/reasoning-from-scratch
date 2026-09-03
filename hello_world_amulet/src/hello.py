"""Smallest possible Amulet job: prove the cluster runs our code and keeps our output."""

import argparse
import datetime
import os
import platform
import socket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    print("hello world from Amulet")
    print(f"host      {socket.gethostname()}")
    print(f"python    {platform.python_version()}")
    print(f"cwd       {os.getcwd()}")
    print(f"out_dir   {args.out_dir}")
    for key in ("AMLT_OUTPUT_DIR", "AMLT_DATA_DIR", "RANK", "WORLD_SIZE"):
        print(f"env       {key}={os.environ.get(key)}")

    try:
        import torch

        print(f"torch     {torch.__version__}")
        print(f"cuda      {torch.cuda.is_available()} devices={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"gpu {i}     {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("torch     not installed in this image")

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out_file = os.path.join(args.out_dir, "hello.txt")
    with open(out_file, "w") as handle:
        handle.write(f"hello world from {socket.gethostname()} at {stamp}\n")
    print(f"wrote     {out_file}")


if __name__ == "__main__":
    main()
