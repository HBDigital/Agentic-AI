"""Run pipeline for week 2026-02-23."""
import subprocess, sys
result = subprocess.run(
    [sys.executable, "main.py", "--week-start", "2026-02-23", "--train"],
    cwd=r"D:\Git\Rebookingagent\GCC_Hack"
)
sys.exit(result.returncode)
