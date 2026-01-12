#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys


def load_wordlist(path):
    words = [w.strip() for w in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if w.strip()]
    if len(words) != 2048:
        raise SystemExit(f"Expected 2048 words, got {len(words)}")
    return {w: i for i, w in enumerate(words)}


def format_perm_words(indices):
    joined = ", ".join(str(i) for i in indices)
    return f"__constant ushort PERM_WORDS[12] = {{{joined}}};"


def main():
    parser = argparse.ArgumentParser(description="Update PERM_WORDS in int_to_address.cl")
    parser.add_argument("--kernel", default="cl/int_to_address.cl")
    parser.add_argument("--wordlist", default="bip39_wordlist.txt")
    parser.add_argument("--words", required=True, help="Space-separated 12 words")
    parser.add_argument("--verbose", action="store_true", help="Print before/after PERM_WORDS line")
    args = parser.parse_args()

    word_map = load_wordlist(args.wordlist)
    words = args.words.strip().split()
    if len(words) != 12:
        raise SystemExit(f"Expected 12 words, got {len(words)}")
    try:
        indices = [word_map[w] for w in words]
    except KeyError as e:
        print(f"Unknown word in list: {e.args[0]}", file=sys.stderr)
        raise SystemExit(2)

    kernel_path = pathlib.Path(args.kernel)
    text = kernel_path.read_text(encoding="utf-8")
    # Match the PERM_WORDS line (allow optional leading whitespace)
    pattern = r"^\s*__constant ushort PERM_WORDS\[12\] = \{.*?\};"
    if args.verbose:
        import re as _re
        m = _re.search(pattern, text, flags=_re.MULTILINE)
        if m:
            print("Before:", m.group(0))
        else:
            print("Before: not found")
    replacement = format_perm_words(indices)
    new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"Failed to find PERM_WORDS line to replace in {args.kernel}")
    if args.verbose:
        import re as _re
        m = _re.search(pattern, new_text, flags=_re.MULTILINE)
        if m:
            print("After :", m.group(0))
    kernel_path.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    main()
