#!/usr/bin/env python3
import argparse
import subprocess
import time


CURRENT_WORDS = [
    "asset",
    "basket",
    "capital",
    "execute",
    "gauge",
    "improve",
    "pair",
    "price",
    "require",
    "sell",
    "share",
    "trend",
]

DEFAULT_ALTS = {
    "execute": ["excuse", "exotic", "expand", "explain"],
    "asset": ["assist", "assume", "asthma"],
    "pair": ["pair", "paint", "panel", "paper", "park", "pine"],
    "sell": ["self", "seminar", "senior", "sense", "sentence"],
    "gauge": ["game", "gap", "garage", "garden", "garlic", "gas"],
}


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd)


def build_phrase(words):
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description="Run one-word substitution sweeps.")
    parser.add_argument("--bin", default="./target/release/bip39-solver-gpu")
    parser.add_argument("--word", help="Limit to a single word key in DEFAULT_ALTS")
    parser.add_argument("--gpu-stats", default="5")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-file", default="tools/substitution_progress.log")
    parser.add_argument("--wordlist", default="../bip39_wordlist.txt")
    parser.add_argument("--kernel", default="cl/int_to_address.cl")
    args = parser.parse_args()

    if args.word and args.word not in DEFAULT_ALTS:
        raise SystemExit(f"Unknown word '{args.word}' (available: {', '.join(DEFAULT_ALTS)})")

    targets = {args.word: DEFAULT_ALTS[args.word]} if args.word else DEFAULT_ALTS

    for word, alts in targets.items():
        if word not in CURRENT_WORDS:
            print(f"[WARN] Word '{word}' not in current phrase, skipping")
            continue
        idx = CURRENT_WORDS.index(word)
        for alt in alts:
            words = CURRENT_WORDS[:]
            words[idx] = alt
            phrase = build_phrase(words)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(args.progress_file, "a", encoding="utf-8") as f:
                f.write(f"{ts} START {word} -> {alt} | {phrase}\n")

            update_cmd = [
                "python3",
                "tools/update_perm_words.py",
                f"--kernel={args.kernel}",
                f"--wordlist={args.wordlist}",
                f"--words={phrase}",
            ]
            update_res = run(update_cmd)
            if update_res.returncode == 2:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(args.progress_file, "a", encoding="utf-8") as f:
                    f.write(f"{ts} SKIP {word} -> {alt} (not in BIP39)\n")
                continue
            if update_res.returncode != 0:
                raise SystemExit(update_res.returncode)

            cmd = [
                args.bin,
                f"--device-index={args.device_index}",
                f"--shard-count={args.shard_count}",
                f"--shard-index={args.shard_index}",
                f"--gpu-stats={args.gpu_stats}",
            ]
            result = run(cmd)

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(args.progress_file, "a", encoding="utf-8") as f:
                status = "DONE" if result.returncode == 0 else "FAIL"
                f.write(f"{ts} {status} {word} -> {alt}\n")
            if result.returncode != 0:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
