"""Verify that all \\includegraphics and \\input paths in a LaTeX file exist."""
import re
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "docs/final.tex"
src = open(path, encoding="utf-8").read()

imgs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src)
inputs = re.findall(r"\\input\{([^}]+)\}", src)

missing = 0

print("Images referenced:")
for i in imgs:
    p = os.path.join("visualizations", i)
    ok = os.path.exists(p)
    print(f"  {'OK' if ok else 'MISSING'}: {p}")
    if not ok:
        missing += 1

print("\nInput files:")
for i in inputs:
    p = i.replace("../", "")
    ok = os.path.exists(p)
    print(f"  {'OK' if ok else 'MISSING'}: {p}")
    if not ok:
        missing += 1

sys.exit(0 if missing == 0 else 1)
