"""Run every verification script and store its output in expected_output/.

Usage:  python run_all.py            (runs everything, about two minutes)
        python run_all.py --check    (also compares with the stored outputs)
"""
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "expected_output"
HELPERS = {"curvature.py", "forms.py"}


def scripts():
    for folder in [ROOT / "scripts", ROOT / "scripts" / "corner"]:
        for p in sorted(folder.glob("*.py")):
            if p.name not in HELPERS:
                yield p


def main():
    check = "--check" in sys.argv
    OUT.mkdir(exist_ok=True)
    failures = []
    for p in scripts():
        t0 = time.time()
        res = subprocess.run([sys.executable, p.name], cwd=p.parent,
                             capture_output=True, text=True)
        dt = time.time() - t0
        name = p.relative_to(ROOT / "scripts").as_posix().replace("/", "__")
        target = OUT / (name[:-3] + ".txt")
        status = "ok" if res.returncode == 0 else "ERROR"
        if res.returncode != 0:
            failures.append(p.name)
            print(res.stderr)
        elif check and target.exists() and target.read_text() != res.stdout:
            status = "ok (output differs from stored, see diff)"
        if not check:
            target.write_text(res.stdout)
        print(f"{status:>8}  {dt:6.1f} s  {p.relative_to(ROOT)}")
    print("\nall scripts ran successfully" if not failures else f"\nfailed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
