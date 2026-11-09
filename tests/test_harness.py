"""python tests/test_harness.py  -- stock plugins must pass, fixtures must fail."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.config import load_config, STOCK_DIR
from core.harness import test_plugin

cfg = load_config()
ok = True
for p in sorted(STOCK_DIR.glob("*.py")):
    t = time.time(); r = test_plugin(p, cfg)
    print(f"{'PASS' if r['passed'] else 'FAIL'} stock/{p.name} {time.time()-t:.1f}s {r['info']} {r['failures'][:3]}")
    ok &= r["passed"]
for p in sorted((ROOT / "tests" / "fixtures").glob("bad_*.py")):
    r = test_plugin(p, cfg)
    print(f"{'FAIL(expected)' if not r['passed'] else 'PASS(WRONG!)'} {p.name}: {r['failures'][:2]}")
    ok &= not r["passed"]
print("ALL OK" if ok else "HARNESS TEST FAILED")
sys.exit(0 if ok else 1)
