"""
Master orchestrator: runs Phases 1-7 sequentially via subprocess.
Each phase is isolated — failure in one doesn't kill others.
Writes SUMMARY.md at the end.
"""
import os, sys, time, subprocess
from pathlib import Path

HERE = Path(__file__).parent
PY = '/home/s20204116/GEN_AI_GTG/.venv/bin/python3'

SUMMARY = HERE / 'SUMMARY.md'
LOG = HERE / 'run_log.txt'

def write_log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [MASTER] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def run_phase(name, script):
    """Run a phase script via subprocess. Return (ok, elapsed_sec)."""
    write_log(f"=== START {name} ===")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PY, str(HERE / script)],
            cwd=str(HERE),
            timeout=14*3600,  # 14h max per phase
            capture_output=False,
        )
        elapsed = time.time() - t0
        ok = (proc.returncode == 0)
        write_log(f"=== END {name} ({elapsed:.0f}s, returncode={proc.returncode}) ===")
        return ok, elapsed
    except subprocess.TimeoutExpired:
        write_log(f"=== TIMEOUT {name} after 14h ===")
        return False, time.time() - t0
    except Exception as e:
        write_log(f"=== ERROR {name}: {e} ===")
        return False, time.time() - t0


def main():
    write_log("=" * 70)
    write_log("MASTER ORCHESTRATOR — Paper Experiment (7 phases)")
    write_log("=" * 70)

    phases = [
        ("Phase 1: Train 3 models",     "run_phase1.py"),
        ("Phase 2: Val performance",    "run_phase2.py"),
        ("Phase 3: Common + Adv",       "run_phase3.py"),
        ("Phase 4: Purification",       "run_phase4.py"),
        ("Phase 5: Analysis",           "run_phase5.py"),
        ("Phase 6: Epsilon",            "run_phase6.py"),
        ("Phase 7: Test set",           "run_phase7.py"),
    ]

    results = []
    t_total = time.time()
    for name, script in phases:
        ok, elapsed = run_phase(name, script)
        results.append({'phase': name, 'ok': ok, 'elapsed_sec': elapsed})

    total = time.time() - t_total

    # SUMMARY.md
    lines = ["# Paper Experiment Summary\n"]
    lines.append(f"Total elapsed: {total/3600:.1f}h")
    lines.append("")
    lines.append("| Phase | Status | Elapsed |")
    lines.append("|---|---|---|")
    for r in results:
        status = "✅" if r['ok'] else "❌"
        lines.append(f"| {r['phase']} | {status} | {r['elapsed_sec']/3600:.2f}h |")
    lines.append("")
    lines.append("## Output paths to check (priority order)")
    lines.append("")
    lines.append("1. **SUMMARY.md** (this file)")
    lines.append("2. `analysis/main_comparison.csv` — 6-model main results")
    lines.append("3. `analysis/paper_results_report.md` — narrative")
    lines.append("4. `test_evaluation/test_comparison.csv` — final paper numbers")
    lines.append("5. `test_evaluation/test_report.md` — test set narrative")
    lines.append("6. `run_log.txt` — full log (check for FAIL/WARNING)")
    SUMMARY.write_text('\n'.join(lines))
    write_log(f"SUMMARY.md written. Total: {total/3600:.1f}h")


if __name__ == '__main__':
    main()
