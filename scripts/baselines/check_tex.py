"""Quick sanity check of LaTeX brace and environment balance."""
import re
from collections import Counter
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "docs/final.tex"
src = open(path, encoding="utf-8").read()

print(f"file: {path}")
print(f"open braces:  {src.count('{')}")
print(f"close braces: {src.count('}')}")

b = Counter(re.findall(r"\\begin\{([A-Za-z*]+)\}", src))
e = Counter(re.findall(r"\\end\{([A-Za-z*]+)\}", src))

ok = True
for k in sorted(set(list(b.keys()) + list(e.keys()))):
    if b[k] != e[k]:
        print(f"  UNBALANCED env: {k} begin={b[k]} end={e[k]}")
        ok = False

print(f"total begin: {sum(b.values())}")
print(f"total end:   {sum(e.values())}")
print(f"ENV OK: {ok}")
