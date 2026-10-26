"""Authorized S12 source workflow, including conditional preregistered confirmation."""
import json
import subprocess
import sys
from core import HERE, ROOT
from run_matrix import run


def script(name, *args):
    subprocess.run([sys.executable, str(HERE / name), *args], cwd=ROOT, check=True)


def commit_results(message):
    subprocess.run(["git", "add", "--", "12-spectral-validation"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)


def main():
    if not (HERE / "source_diagnostics.json").exists():
        script("source_diagnostics.py")
    run("smoke", ["0", "1", "2", "3"], 1)
    run("pilot", ["0", "1", "2", "3"], 2)
    script("report.py", "--phase", "pilot")
    commit_results("S12: record source-only pilot and freeze conditional confirmation")
    decision = json.loads((HERE / "decision_pilot.json").read_text())
    if decision["admitted"]:
        run("confirmation", ["0", "1", "2", "3"], 2)
        script("report.py", "--phase", "confirmation")
        commit_results("S12: record independent source confirmation")
    print("S12 authorized source workflow finished; target access remains disabled.", flush=True)


if __name__ == "__main__":
    main()
