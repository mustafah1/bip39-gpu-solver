# BIP39 Solver GPU

Lightweight GPU solver to scan BIP39 permutations for a target address. Bundles the BIP39 wordlist and several strategy wordlists for quick deployment on Kaggle (dual T4) or local GPUs.

## Build (local)
```bash
cargo build --release
```

## Run (local)
```bash
./target/release/bip39-solver-gpu --gpu-stats=5
```

## Kaggle quickstart (T4 x2)
All required files are in the repo (`bip39_wordlist.txt`, `gpu_wordlists/strategy*.txt`).

```bash
%%bash
cd /kaggle/working/bip39-gpu-solver
git fetch --all
git reset --hard origin/master

# Install Rust (one-time per session) and build
curl -sSf https://sh.rustup.rs | sh -s -- -y
/root/.cargo/bin/cargo build --release

# Patch kernel with 12 valid BIP39 placeholders.
# Permutations come from the strategy list you choose.
python3 tools/update_perm_words.py \
  --wordlist bip39_wordlist.txt \
  --words "basket capital execute gauge improve interest pine price asset risk market common"
```

Then start the dual-GPU loop:
```python
import os, subprocess, threading, time
os.chdir("/kaggle/working/bip39-gpu-solver")

def launch(cmd, name):
    log = open(f"{name}.log", "w", buffering=1)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return p, log.name

def tail(path, prefix, p0, p1):
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if line:
                print(f"[{prefix}] {line}", end="")
            else:
                if p0.poll() is not None and p1.poll() is not None:
                    break
                time.sleep(0.25)

while True:
    cmd0 = ["./target/release/bip39-solver-gpu", "--device-index=0", "--shard-count=2", "--shard-index=0", "--gpu-stats=5"]
    cmd1 = ["./target/release/bip39-solver-gpu", "--device-index=1", "--shard-count=2", "--shard-index=1", "--gpu-stats=5"]
    p0, log0 = launch(cmd0, "gpu0")
    p1, log1 = launch(cmd1, "gpu1")
    t0 = threading.Thread(target=tail, args=(log0, "GPU0", p0, p1), daemon=True)
    t1 = threading.Thread(target=tail, args=(log1, "GPU1", p0, p1), daemon=True)
    t0.start(); t1.start()
    while p0.poll() is None or p1.poll() is None:
        time.sleep(1)
    print("Strategy run complete. Restarting in 10s...")
    time.sleep(10)
```

### Strategy wordlists
Bundled under `gpu_wordlists/`:
- `strategy1_top_frequency.txt`
- `strategy2_anomaly_pages.txt` (recommended)
- `strategy3_all_candidates.txt`
- `strategy4_frequency_weighted.txt`
- `strategy5_anomaly_pages.txt`
- `strategy5_top_frequency.txt`

To switch strategies, rerun `tools/update_perm_words.py` with any 12 valid BIP39 placeholders (only for seeding) and restart the solver. The permutations come from the strategy list, not the placeholder seed.
