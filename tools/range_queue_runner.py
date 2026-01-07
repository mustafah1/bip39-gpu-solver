#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time


def parse_ranges(range_args):
    ranges = []
    for item in range_args:
        if ":" not in item:
            raise ValueError(f"Invalid range '{item}', expected start:end")
        start_s, end_s = item.split(":", 1)
        ranges.append((int(start_s), int(end_s)))
    return ranges


def read_ranges_file(path):
    ranges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid range line '{line}', expected start:end")
            start_s, end_s = line.split(":", 1)
            ranges.append((int(start_s), int(end_s)))
    return ranges


def main():
    parser = argparse.ArgumentParser(description="Run GPU solver over a queue of ranges.")
    parser.add_argument("--bin", default="./target/release/bip39-solver-gpu")
    parser.add_argument("--range", dest="ranges", action="append", default=[], help="Range start:end")
    parser.add_argument("--ranges-file", help="File with one start:end per line")
    parser.add_argument("--gpu-stats", nargs="?", const="5", help="Enable GPU stats polling (seconds)")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--progress-file", default="tools/ranges_progress.log")
    parser.add_argument("--update-ranges-file", action="store_true", help="Remove completed ranges from ranges file")
    args = parser.parse_args()

    ranges = []
    if args.ranges_file:
        ranges.extend(read_ranges_file(args.ranges_file))
    if args.ranges:
        ranges.extend(parse_ranges(args.ranges))

    if not ranges:
        print("No ranges provided. Use --range start:end or --ranges-file", file=sys.stderr)
        sys.exit(2)

    def log_progress(status, start, end):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(args.progress_file, "a", encoding="utf-8") as f:
            f.write(f"{ts} {status} {start}:{end}\n")

    def remove_range_from_file(start, end):
        if not args.ranges_file:
            return
        kept = []
        with open(args.ranges_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    kept.append(line)
                    continue
                if line == f"{start}:{end}":
                    continue
                kept.append(line)
        tmp_path = args.ranges_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for line in kept:
                if line:
                    f.write(line + "\n")
        os.replace(tmp_path, args.ranges_file)

    for start, end in ranges:
        cmd = [
            args.bin,
            f"--start={start}",
            f"--end={end}",
            f"--shard-count={args.shard_count}",
            f"--shard-index={args.shard_index}",
            f"--device-index={args.device_index}",
        ]
        if args.gpu_stats is not None:
            cmd.append(f"--gpu-stats={args.gpu_stats}")
        print("$", " ".join(cmd), flush=True)
        log_progress("START", start, end)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            log_progress("FAIL", start, end)
            sys.exit(result.returncode)
        log_progress("DONE", start, end)
        if args.update_ranges_file:
            remove_range_from_file(start, end)


if __name__ == "__main__":
    main()
